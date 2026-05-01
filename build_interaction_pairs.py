import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import time
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from PIL import Image
from transformers import pipeline
from tqdm import tqdm


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
DEFAULT_MODEL_ID = "Qwen/Qwen3.5-4B"


def log(message: str) -> None:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


@dataclass
class ImageInfo:
    category: str
    fsn: str
    image_id: str
    path: str
    width: int
    height: int
    sha256: str


@dataclass
class PairRow:
    pair_uid: str
    category: str
    fsn: str
    input_images: list[str]
    target_image: str
    edit_instruction: str


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def compute_sha256(file_path: str) -> str:
    digest = hashlib.sha256()
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def sanitize_segment(segment: str) -> str:
    segment = str(segment or "").strip().lower()
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in segment) or "unknown"


def write_jsonl_row(file_path: str, row: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_jsonl_records(jsonl_path: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                records.append(value)
    return records


def guess_extension_from_url(url: str) -> str:
    clean = str(url).split("?", 1)[0]
    suffix = Path(clean).suffix.lower()
    return suffix if suffix in IMAGE_EXTENSIONS else ".jpg"


def download_images_for_fsn(
    image_url: dict[str, Any],
    images_dir: str,
    category: str,
    fsn: str,
    timeout_sec: int = 30,
) -> list[dict[str, Any]]:
    os.makedirs(images_dir, exist_ok=True)
    anomalies: list[dict[str, Any]] = []
    for image_key, image_source in sorted(image_url.items(), key=lambda x: str(x[0])):
        image_key = str(image_key)
        source = str(image_source or "").strip()
        if not source:
            anomalies.append(
                {
                    "type": "invalid_image_url",
                    "category": category,
                    "fsn": fsn,
                    "image_key": image_key,
                    "source": source,
                }
            )
            continue

        ext = guess_extension_from_url(source)
        output_path = os.path.join(images_dir, f"{image_key}{ext}")
        try:
            request = urllib.request.Request(source, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(request, timeout=timeout_sec) as response:
                content = response.read()
            with open(output_path, "wb") as f:
                f.write(content)
        except Exception as exc:  # noqa: BLE001
            anomalies.append(
                {
                    "type": "image_download_failed",
                    "category": category,
                    "fsn": fsn,
                    "image_key": image_key,
                    "source": source,
                    "reason": str(exc),
                }
            )
    return anomalies


def has_image_shape(file_path: str) -> tuple[bool, Optional[int], Optional[int], Optional[str]]:
    try:
        with Image.open(file_path) as img:
            width, height = img.size
        return True, width, height, None
    except Exception as exc:  # noqa: BLE001
        return False, None, None, str(exc)


def resolve_image_path(images_dir: str, image_key: str) -> Optional[str]:
    for ext in (".jpg", ".jpeg", ".png", ".webp"):
        candidate = os.path.join(images_dir, f"{image_key}{ext}")
        if os.path.isfile(candidate):
            return candidate

    matches = sorted(Path(images_dir).glob(f"{image_key}.*"))
    for match in matches:
        if match.is_file():
            return str(match)
    return None


def gather_images_for_fsn(
    dataset_root: str,
    category: str,
    fsn: str,
    image_url: dict[str, Any],
) -> tuple[list[ImageInfo], list[dict[str, Any]]]:
    images_dir = os.path.join(dataset_root, category, fsn, "images")
    anomalies: list[dict[str, Any]] = []
    collected: list[ImageInfo] = []
    if not os.path.isdir(images_dir):
        anomalies.append(
            {
                "type": "missing_images_dir",
                "category": category,
                "fsn": fsn,
                "path": images_dir,
            }
        )
        return collected, anomalies

    for image_key, image_source in sorted(image_url.items(), key=lambda x: str(x[0])):
        image_key = str(image_key)
        image_path = resolve_image_path(images_dir, image_key)
        if image_path is None:
            anomalies.append(
                {
                    "type": "missing_image_for_key",
                    "category": category,
                    "fsn": fsn,
                    "image_key": image_key,
                    "source": str(image_source),
                    "images_dir": images_dir,
                }
            )
            continue

        suffix = Path(image_path).suffix.lower()
        if suffix and suffix not in IMAGE_EXTENSIONS:
            continue

        ok, width, height, error = has_image_shape(image_path)
        if not ok:
            anomalies.append(
                {
                    "type": "corrupt_image",
                    "category": category,
                    "fsn": fsn,
                    "image": os.path.abspath(image_path),
                    "reason": error,
                }
            )
            continue

        sha = compute_sha256(image_path)

        collected.append(
            ImageInfo(
                category=category,
                fsn=fsn,
                image_id=image_key,
                path=os.path.abspath(image_path),
                width=width,
                height=height,
                sha256=sha,
            )
        )

    unique_by_hash: dict[str, ImageInfo] = {}
    for image in collected:
        unique_by_hash.setdefault(image.sha256, image)

    if len(unique_by_hash) != len(collected):
        anomalies.append(
            {
                "type": "duplicate_images",
                "category": category,
                "fsn": fsn,
                "unique_count": len(unique_by_hash),
                "total_count": len(collected),
            }
        )
    return list(unique_by_hash.values()), anomalies


def image_source_from_record(image_url: dict[str, Any], image_id: str, fallback_path: str) -> str:
    value = image_url.get(image_id)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return fallback_path


def stable_uid(*parts: str) -> str:
    content = "::".join(parts)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:20]


def build_metadata_summary(metadata: dict[str, Any]) -> str:
    if not metadata:
        return ""
    filtered = {
        "title": metadata.get("title"),
        "vertical": metadata.get("vertical") or metadata.get("cms_vertical"),
        "brand": metadata.get("brand"),
        "attributes": metadata.get("attributes"),
    }
    compact = json.dumps(filtered, ensure_ascii=False)
    return compact


class TransformersVLMClient:
    def __init__(self, model_id: str, use_8bit: bool, log_response_chars: int = 0) -> None:
        self.model_id = model_id
        self.log_response_chars = max(0, int(log_response_chars))
        load_start = time.time()
        log(f"Initializing transformers pipeline for model={model_id}, use_8bit={use_8bit}")
        model_kwargs: dict[str, Any] = {
            "device_map": "auto",
        }
        if use_8bit:
            model_kwargs["load_in_8bit"] = True

        self.pipe = pipeline(
            task="image-text-to-text",
            model=model_id,
            trust_remote_code=True,
            model_kwargs=model_kwargs,
        )
        log(f"Pipeline ready in {time.time() - load_start:.1f}s")

    def _extract_text(self, output: Any) -> str:
        if isinstance(output, list) and output:
            item = output[0]
            if isinstance(item, dict):
                generated = item.get("generated_text")
                if isinstance(generated, str):
                    return generated
                if isinstance(generated, list):
                    for turn in reversed(generated):
                        if isinstance(turn, dict):
                            content = turn.get("content")
                            if isinstance(content, str):
                                return content
                            if isinstance(content, list):
                                for block in content:
                                    if isinstance(block, dict) and isinstance(block.get("text"), str):
                                        return block["text"]
        if isinstance(output, str):
            return output
        return ""

    def generate_text(
        self,
        prompt: str,
        image_paths: list[str],
        op_name: str = "vlm_call",
        max_new_tokens: Optional[int] = 512,
        do_sample: bool = False,
    ) -> str:
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for path in image_paths:
            content.append({"type": "image", "image": path})

        messages = [{"role": "user", "content": content}]
        try:
            generate_kwargs: dict[str, Any] = {
                "text": messages,
                "do_sample": do_sample,
                "return_full_text": False,
            }
            if max_new_tokens is not None:
                generate_kwargs["max_new_tokens"] = max_new_tokens
            output = self.pipe(**generate_kwargs)
            text = self._extract_text(output)
            if not text:
                raise RuntimeError("No text output returned from model response")
            if self.log_response_chars > 0:
                preview = text[: self.log_response_chars].replace("\n", " ")
                suffix = "..." if len(text) > self.log_response_chars else ""
                log(f"{op_name}: raw_response={preview}{suffix}")
            return text
        except Exception as exc:  # noqa: BLE001
            log(f"{op_name}: failed, error={exc}")
            raise


def _has_negated_token(text: str, token_group: str) -> bool:
    pattern = rf"\b(no|not|without)\s+(?:\w+\s+){{0,2}}(?:{token_group})\b"
    return re.search(pattern, text) is not None


def infer_interaction_from_text(raw_text: str) -> Optional[dict[str, Any]]:
    if not raw_text:
        return None

    text = raw_text.lower()
    person_tokens = r"person|human|girl|boy|man|woman|child|kid|lady|guy"
    hand_tokens = r"hand|hands|holding|hold|grip|touch|press|operate|using"

    has_person_signal = re.search(rf"\b({person_tokens})\b", text) is not None
    has_hand_signal = re.search(rf"\b({hand_tokens})\b", text) is not None

    has_person = has_person_signal and not _has_negated_token(text, person_tokens)
    has_hands_interaction = has_hand_signal and not _has_negated_token(text, hand_tokens)
    person_near_product = has_person and (
        "with" in text or "near" in text or "holding" in text or "next to" in text
    )

    if has_hands_interaction:
        interaction_confidence = 0.85
    elif person_near_product:
        interaction_confidence = 0.72
    elif has_person:
        interaction_confidence = 0.62
    else:
        interaction_confidence = 0.20

    return {
        "has_person": bool(has_person),
        "has_hands_interaction": bool(has_hands_interaction),
        "person_near_product": bool(person_near_product),
        "product_match_confidence": 0.75,
        "interaction_confidence": interaction_confidence,
        "explanation": "Derived from non-JSON model response using keyword fallback parser.",
    }


def extract_final_yes_no(raw_text: str) -> Optional[bool]:
    if not raw_text:
        return None

    # Prefer explicit final markers.
    pattern = re.compile(r"(?:final_answer|answer|decision)\s*[:\-]\s*(yes|no)\b", re.IGNORECASE)
    matches = list(pattern.finditer(raw_text))
    if matches:
        return matches[-1].group(1).lower() == "yes"

    # Fallback to last standalone yes/no token in the text.
    yn_matches = re.findall(r"\b(yes|no)\b", raw_text, flags=re.IGNORECASE)
    if yn_matches:
        return yn_matches[-1].lower() == "yes"

    return None


def extract_final_instruction(raw_text: str) -> Optional[str]:
    if not raw_text:
        return None

    m = re.search(r"final_instruction\s*[:\-]\s*(.+)", raw_text, flags=re.IGNORECASE | re.DOTALL)
    if m:
        candidate = re.sub(r"\s+", " ", m.group(1)).strip()
        if candidate:
            return candidate
    return None


def classify_interaction(
    client: TransformersVLMClient,
    image: ImageInfo,
    metadata_summary: str,
) -> dict[str, Any]:
    prompt = (
        "You are a strict image analyzer for data curation.\n"
        "Task: decide whether this image is a valid HUMAN-INTERACTION TARGET for the shown product.\n"
        "Valid when a person is present and either: (a) directly interacting with product, or (b) clearly near/presenting product.\n"
        "First give short analysis in plain text (2-5 bullet points).\n"
        "Then END with exactly one line in this format:\n"
        "FINAL_ANSWER: YES\n"
        "or\n"
        "FINAL_ANSWER: NO\n"
        "No JSON required.\n"
        "Product metadata summary: "
        f"{metadata_summary}"
    )
    raw = client.generate_text(
        prompt=prompt,
        image_paths=[image.path],
        op_name=f"classify_interaction:{image.fsn}:{image.image_id}",
    )
    decision = extract_final_yes_no(raw)
    if decision is None:
        # If the model ignores final-line formatting, fallback to weak NLP parsing.
        fallback = infer_interaction_from_text(raw)
        if fallback is not None:
            log(f"classify_interaction:{image.fsn}:{image.image_id}: missing FINAL_ANSWER; used prose fallback parser")
            return fallback
        preview = raw.replace("\n", " ")
        log(f"classify_interaction:{image.fsn}:{image.image_id}: missing FINAL_ANSWER and fallback failed, preview={preview}")
        raise ValueError(f"Model response not parseable for image {image.path}")

    explanation = raw.replace("\n", " ")
    return {
        "has_person": bool(decision),
        "has_hands_interaction": bool(decision),
        "person_near_product": bool(decision),
        "product_match_confidence": 0.8 if decision else 0.5,
        "interaction_confidence": 0.85 if decision else 0.15,
        "explanation": f"Parsed FINAL_ANSWER from model output: {'YES' if decision else 'NO'} | {explanation}",
    }


def generate_edit_instruction(
    client: TransformersVLMClient,
    fsn: str,
    category: str,
    input_images: list[ImageInfo],
    target_image: ImageInfo,
    metadata_summary: str,
) -> str:
    instruction_prompt = (
        "Write one precise image-edit instruction that transforms the non-target product views into the target human-interaction view.\n"
        "The instruction may be long if needed to capture all important transformations accurately.\n"
        "Include all relevant changes visible in target: human presence and interaction, pose, framing/viewpoint, background, lighting/style, and object placement, while preserving product identity.\n"
        "Do not invent brand/spec details not visible.\n"
        "Output format rules:\n"
        "1) Output exactly one line.\n"
        "2) Start with: FINAL_INSTRUCTION: \n"
        "3) No bullets, no analysis, no planning text.\n"
        f"FSN: {fsn}; category: {category}; metadata: {metadata_summary}\n"
    )
    print(instruction_prompt)
    image_paths = [item.path for item in input_images] + [target_image.path]
    raw = client.generate_text(
        prompt=instruction_prompt,
        image_paths=image_paths,
        op_name=f"generate_edit_instruction:{fsn}:{target_image.image_id}",
        max_new_tokens=None,
        do_sample=False,
    )
    instruction = extract_final_instruction(raw)
    if not instruction:
        preview = raw[:500].replace("\n", " ")
        log(f"generate_edit_instruction:{fsn}:{target_image.image_id}: unparseable response preview={preview}")
        raise ValueError(f"Instruction parse failed for fsn={fsn}, target={target_image.image_id}")
    return instruction.strip()


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build paired dataset rows (input_images, edit_instruction, target_image) from furniture dataset."
    )
    parser.add_argument(
        "--dataset-root",
        default=None,
        help="Optional temporary image workspace root. If omitted, uses <out-dir>/_tmp_images.",
    )
    parser.add_argument("--records-jsonl", default=None, help="Path to source JSONL with fsn/vertical/image_url/attributes")
    parser.add_argument("--out-dir", required=True, help="Path for generated JSONL outputs")
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help="Transformers model id (default: Qwen/Qwen3.5-35B-A3B).",
    )
    parser.add_argument("--use-8bit", action="store_true", help="Enable 8-bit quantization")
    parser.add_argument("--max-fsns", type=int, default=None, help="Optional limit for pilot runs")
    parser.add_argument(
        "--log-response-chars",
        type=int,
        default=0,
        help="Log first N characters of every model response (0 disables, recommended 300-800 for debugging).",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=10,
        help="Emit a progress log every N FSNs (default: 10)",
    )
    parser.add_argument(
        "--min-interaction-confidence",
        type=float,
        default=0.70,
        help="Minimum interaction confidence to accept an image as target candidate",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Total worker shards for parallel runs (use 8 for 8-GPU data parallel runs).",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Current shard index in [0, num_shards-1].",
    )
    parser.add_argument(
        "--merge-shards",
        action="store_true",
        help="Merge shard outputs into single files in out-dir and exit.",
    )
    return parser.parse_args(argv)


def get_output_paths(out_dir: str, num_shards: int, shard_index: int) -> dict[str, str]:
    suffix = f".shard{shard_index:02d}" if num_shards > 1 else ""
    return {
        "accepted": os.path.join(out_dir, f"accepted_pairs{suffix}.jsonl"),
        "skipped_fsn": os.path.join(out_dir, f"skipped_fsns{suffix}.jsonl"),
        "skipped_pairs": os.path.join(out_dir, f"skipped_pairs{suffix}.jsonl"),
        "anomalies": os.path.join(out_dir, f"anomalies{suffix}.jsonl"),
        "summary": os.path.join(out_dir, f"stage_summary{suffix}.json"),
    }


def _iter_lines(file_path: str) -> list[str]:
    if not os.path.isfile(file_path):
        return []
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f if line.strip()]


def merge_shard_outputs(out_dir: str, num_shards: int) -> dict[str, Any]:
    merged_paths = get_output_paths(out_dir, num_shards=1, shard_index=0)
    shard_paths = [get_output_paths(out_dir, num_shards=num_shards, shard_index=i) for i in range(num_shards)]

    for key in ("accepted", "skipped_fsn", "skipped_pairs", "anomalies"):
        if os.path.isfile(merged_paths[key]):
            os.remove(merged_paths[key])

    seen_pair_uids: set[str] = set()
    merged_counts = {
        "accepted_pairs": 0,
        "skipped_fsns": 0,
        "skipped_pairs": 0,
        "anomalies": 0,
        "missing_shard_files": 0,
    }

    for shard_idx, paths in enumerate(shard_paths):
        accepted_rows = _iter_lines(paths["accepted"])
        skipped_fsn_rows = _iter_lines(paths["skipped_fsn"])
        skipped_pairs_rows = _iter_lines(paths["skipped_pairs"])
        anomaly_rows = _iter_lines(paths["anomalies"])

        if not accepted_rows and not skipped_fsn_rows and not skipped_pairs_rows and not anomaly_rows:
            merged_counts["missing_shard_files"] += 1
            log(f"merge: shard {shard_idx} appears empty or missing")

        for line in accepted_rows:
            try:
                obj = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            pair_uid = str(obj.get("pair_uid", ""))
            if pair_uid and pair_uid in seen_pair_uids:
                continue
            if pair_uid:
                seen_pair_uids.add(pair_uid)
            write_jsonl_row(merged_paths["accepted"], obj)
            merged_counts["accepted_pairs"] += 1

        for line in skipped_fsn_rows:
            try:
                write_jsonl_row(merged_paths["skipped_fsn"], json.loads(line))
                merged_counts["skipped_fsns"] += 1
            except Exception:  # noqa: BLE001
                continue

        for line in skipped_pairs_rows:
            try:
                write_jsonl_row(merged_paths["skipped_pairs"], json.loads(line))
                merged_counts["skipped_pairs"] += 1
            except Exception:  # noqa: BLE001
                continue

        for line in anomaly_rows:
            try:
                write_jsonl_row(merged_paths["anomalies"], json.loads(line))
                merged_counts["anomalies"] += 1
            except Exception:  # noqa: BLE001
                continue

    merged_summary = {
        "out_dir": os.path.abspath(out_dir),
        "num_shards": num_shards,
        **merged_counts,
        "completed_at": utc_now_iso(),
    }
    with open(merged_paths["summary"], "w", encoding="utf-8") as f:
        json.dump(merged_summary, f, ensure_ascii=False, indent=2)

    log(
        "merge completed: "
        f"accepted_pairs={merged_counts['accepted_pairs']} "
        f"skipped_fsns={merged_counts['skipped_fsns']} "
        f"skipped_pairs={merged_counts['skipped_pairs']} "
        f"anomalies={merged_counts['anomalies']}"
    )
    return merged_summary


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    out_dir = os.path.abspath(args.out_dir)
    records_jsonl = os.path.abspath(args.records_jsonl) if args.records_jsonl else None
    os.makedirs(out_dir, exist_ok=True)

    if args.num_shards < 1:
        print("--num-shards must be >= 1", file=sys.stderr)
        return 1
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        print("--shard-index must satisfy 0 <= shard_index < num_shards", file=sys.stderr)
        return 1

    if args.merge_shards:
        merge_shard_outputs(out_dir=out_dir, num_shards=args.num_shards)
        return 0

    if not records_jsonl:
        print("--records-jsonl is required unless --merge-shards is provided", file=sys.stderr)
        return 1
    if not os.path.isfile(records_jsonl):
        print(f"Records JSONL not found: {records_jsonl}", file=sys.stderr)
        return 1

    image_workspace_root = os.path.abspath(args.dataset_root) if args.dataset_root else os.path.join(out_dir, "_tmp_images")
    shard_image_root = os.path.join(image_workspace_root, f"shard_{args.shard_index:02d}")
    os.makedirs(shard_image_root, exist_ok=True)

    output_paths = get_output_paths(out_dir, args.num_shards, args.shard_index)
    accepted_path = output_paths["accepted"]
    skipped_fsn_path = output_paths["skipped_fsn"]
    skipped_pairs_path = output_paths["skipped_pairs"]
    anomalies_path = output_paths["anomalies"]
    summary_path = output_paths["summary"]

    # Start fresh output files for deterministic reruns.
    for path in (accepted_path, skipped_fsn_path, skipped_pairs_path, anomalies_path):
        if os.path.isfile(path):
            os.remove(path)

    log(f"Loading Transformers VLM model: {args.model_id}")
    client = TransformersVLMClient(
        model_id=args.model_id,
        use_8bit=bool(args.use_8bit),
        log_response_chars=args.log_response_chars,
    )

    all_records = load_jsonl_records(records_jsonl)
    shard_records = [item for idx, item in enumerate(all_records) if idx % args.num_shards == args.shard_index]
    if args.max_fsns is not None:
        shard_records = shard_records[: args.max_fsns]
    log(
        f"Discovered total_records={len(all_records)}, "
        f"shard_records={len(shard_records)}, shard_index={args.shard_index}/{args.num_shards}"
    )

    summary: dict[str, Any] = {
        "image_workspace_root": image_workspace_root,
        "records_jsonl": records_jsonl,
        "out_dir": out_dir,
        "model_id": args.model_id,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "processed_fsns": 0,
        "accepted_pairs": 0,
        "skipped_fsns": 0,
        "skipped_pairs": 0,
        "anomalies": 0,
        "started_at": utc_now_iso(),
    }

    run_start = time.time()
    for record in tqdm(shard_records, desc="Curating FSNs"):
        fsn = str(record.get("fsn") or "").strip()
        vertical_raw = str(record.get("vertical") or "unknown")
        category = sanitize_segment(vertical_raw)
        image_url = record.get("image_url")

        summary["processed_fsns"] += 1
        fsn_start = time.time()
        log(f"FSN start: {category}/{fsn}")

        if not fsn or not isinstance(image_url, dict) or not image_url:
            write_jsonl_row(
                skipped_fsn_path,
                {
                    "category": category,
                    "fsn": fsn,
                    "reason": "invalid_record_missing_fsn_or_image_url",
                },
            )
            summary["skipped_fsns"] += 1
            log(f"{category}/{fsn}: skipped (invalid_record_missing_fsn_or_image_url)")
            continue

        if args.log_every > 0 and summary["processed_fsns"] % args.log_every == 0:
            log(
                "Progress checkpoint: "
                f"processed={summary['processed_fsns']} "
                f"accepted_pairs={summary['accepted_pairs']} "
                f"skipped_fsns={summary['skipped_fsns']} "
                f"skipped_pairs={summary['skipped_pairs']} "
                f"anomalies={summary['anomalies']}"
            )

        attributes = record.get("attributes") if isinstance(record.get("attributes"), dict) else {}
        metadata = {
            "title": attributes.get("title"),
            "vertical": vertical_raw,
            "brand": attributes.get("brand"),
            "attributes": attributes,
        }
        metadata_summary = build_metadata_summary(metadata)

        fsn_temp_dir = os.path.join(shard_image_root, category, fsn)
        fsn_images_dir = os.path.join(fsn_temp_dir, "images")

        try:
            download_anomalies = download_images_for_fsn(
                image_url=image_url,
                images_dir=fsn_images_dir,
                category=category,
                fsn=fsn,
            )
            for row in download_anomalies:
                write_jsonl_row(anomalies_path, row)
                summary["anomalies"] += 1

            images, image_anomalies = gather_images_for_fsn(
                dataset_root=shard_image_root,
                category=category,
                fsn=fsn,
                image_url=image_url,
            )
            log(f"{category}/{fsn}: valid_images={len(images)}, anomalies_found={len(image_anomalies)}")
            for row in image_anomalies:
                write_jsonl_row(anomalies_path, row)
                summary["anomalies"] += 1

            if len(images) < 2:
                write_jsonl_row(
                    skipped_fsn_path,
                    {
                        "category": category,
                        "fsn": fsn,
                        "reason": "insufficient_valid_images",
                        "valid_image_count": len(images),
                    },
                )
                summary["skipped_fsns"] += 1
                log(f"{category}/{fsn}: skipped (insufficient_valid_images)")
                continue

            interaction_results: dict[str, dict[str, Any]] = {}
            target_candidates: list[ImageInfo] = []
            interaction_failures: list[dict[str, Any]] = []

            for image in images:
                try:
                    analysis = classify_interaction(client=client, image=image, metadata_summary=metadata_summary)
                except Exception as exc:  # noqa: BLE001
                    interaction_failures.append(
                        {
                            "type": "interaction_model_failure",
                            "category": category,
                            "fsn": fsn,
                            "image": image.path,
                            "image_id": image.image_id,
                            "details": str(exc),
                        }
                    )
                    log(
                        f"{category}/{fsn}: classify failed on image={image.image_id}; "
                        "continuing with remaining images"
                    )
                    continue

                interaction_results[image.path] = analysis
                has_person = bool(analysis.get("has_person", False))
                has_hands_interaction = bool(analysis.get("has_hands_interaction", False))
                person_near_product = bool(analysis.get("person_near_product", False))
                confidence = float(analysis.get("interaction_confidence", 0.0) or 0.0)

                if has_person and (has_hands_interaction or person_near_product) and confidence >= args.min_interaction_confidence:
                    target_candidates.append(image)

            if interaction_failures:
                for row in interaction_failures:
                    write_jsonl_row(anomalies_path, row)
                    summary["anomalies"] += 1

            if not interaction_results:
                write_jsonl_row(
                    skipped_fsn_path,
                    {
                        "category": category,
                        "fsn": fsn,
                        "reason": "interaction_model_failure_all_images",
                        "failed_images": [row["image_id"] for row in interaction_failures],
                    },
                )
                summary["skipped_fsns"] += 1
                log(f"{category}/{fsn}: skipped (interaction_model_failure_all_images)")
                continue

            log(f"{category}/{fsn}: target_candidates={len(target_candidates)}")

            if not target_candidates:
                write_jsonl_row(
                    skipped_fsn_path,
                    {
                        "category": category,
                        "fsn": fsn,
                        "reason": "no_human_interaction_targets",
                        "image_count": len(images),
                    },
                )
                summary["skipped_fsns"] += 1
                log(f"{category}/{fsn}: skipped (no_human_interaction_targets)")
                continue

            target_paths = {target.path for target in target_candidates}
            non_target_images = [image for image in images if image.path not in target_paths]
            if not non_target_images:
                write_jsonl_row(
                    skipped_fsn_path,
                    {
                        "category": category,
                        "fsn": fsn,
                        "reason": "all_images_are_targets_no_input_images",
                        "target_count": len(target_candidates),
                        "image_count": len(images),
                    },
                )
                summary["skipped_fsns"] += 1
                log(f"{category}/{fsn}: skipped (all_images_are_targets_no_input_images)")
                continue

            for target in target_candidates:
                input_images = non_target_images
                target_source = image_source_from_record(image_url, target.image_id, target.path)
                if not input_images:
                    write_jsonl_row(
                        skipped_pairs_path,
                        {
                            "category": category,
                            "fsn": fsn,
                            "target_image": target_source,
                            "reason": "no_valid_input_images",
                        },
                    )
                    summary["skipped_pairs"] += 1
                    log(f"{category}/{fsn}: target={target.image_id} skipped (no_valid_input_images)")
                    continue

                try:
                    instruction = generate_edit_instruction(
                        client=client,
                        fsn=fsn,
                        category=category,
                        input_images=input_images,
                        target_image=target,
                        metadata_summary=metadata_summary,
                    )
                except Exception as exc:  # noqa: BLE001
                    write_jsonl_row(
                        skipped_pairs_path,
                        {
                            "category": category,
                            "fsn": fsn,
                            "target_image": target_source,
                            "reason": "instruction_generation_failed",
                            "details": str(exc),
                        },
                    )
                    summary["skipped_pairs"] += 1
                    log(f"{category}/{fsn}: target={target.image_id} skipped (instruction_generation_failed)")
                    continue

                pair_uid = stable_uid(
                    category,
                    fsn,
                    target.sha256,
                    "|".join(sorted([img.sha256 for img in input_images])),
                )
                row = PairRow(
                    pair_uid=pair_uid,
                    category=category,
                    fsn=fsn,
                    input_images=[image_source_from_record(image_url, img.image_id, img.path) for img in input_images],
                    target_image=target_source,
                    edit_instruction=instruction,
                )
                write_jsonl_row(accepted_path, asdict(row))
                summary["accepted_pairs"] += 1
                log(
                    f"{category}/{fsn}: target={target.image_id} accepted, "
                    f"inputs={len(input_images)}, pair_uid={pair_uid}"
                )
        finally:
            shutil.rmtree(fsn_temp_dir, ignore_errors=True)

        log(f"FSN end: {category}/{fsn} in {time.time() - fsn_start:.1f}s")

    summary["completed_at"] = utc_now_iso()
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    log(f"Curation completed in {time.time() - run_start:.1f}s")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())