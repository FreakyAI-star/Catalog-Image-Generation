import csv
import os

import torch
from PIL import Image
from diffusers import DiffusionPipeline
from diffusers.quantizers import PipelineQuantizationConfig
from diffusers.utils import load_image
# from diffusers import LongCatImageEditPipeline


OUTPUT_DIR = "generated_images"
CSV_PATH = "product_interaction_prompts.csv"
INPUT_DIR = "downloaded_images"


os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(INPUT_DIR, exist_ok=True)


def main() -> None:
    # Define the pipeline-specific quantization configuration
    quant_config = PipelineQuantizationConfig(
        quant_backend="bitsandbytes_8bit",
        quant_kwargs={"load_in_8bit": True},
    )
    

    # # Load the pipelines
    # print("Loading FireRed pipeline...")
    # fire_pipe = DiffusionPipeline.from_pretrained(
    #     "FireRedTeam/FireRed-Image-Edit-1.0",
    #     torch_dtype=torch.bfloat16,
    #     quantization_config=quant_config,
    #     device_map="cuda",
    # )

    # longcat_pipe = None
    # try:
    #     # Use the latest turbo edit model for LongCat-Image
    #     print("Loading LongCat-Image-Edit pipeline...")
    #     longcat_pipe = LongCatImageEditPipeline.from_pretrained(
    #         "meituan-longcat/LongCat-Image-Edit",
    #         torch_dtype=torch.bfloat16,
    #         quantization_config=quant_config,
    #         device_map="cuda",
    #     )
    # except Exception as e:
    #     print(f"Could not load LongCat-Image-Edit pipeline, skipping LongCat edits: {e}")

    qwen_pipe = None
    try:
        print("Loading Qwen-Image-Edit-2511 pipeline...")
        qwen_pipe = DiffusionPipeline.from_pretrained(
            "Qwen/Qwen-Image-Edit-2511",
            torch_dtype=torch.bfloat16,
            quantization_config=quant_config,
            device_map="cuda",
        )
    except Exception as e:
        print(f"Could not load Qwen-Image-Edit pipeline, skipping Qwen edits: {e}")

    flux_pipe = None
    try:
        print("Loading FLUX.2-klein-base-9B pipeline...")
        flux_pipe = DiffusionPipeline.from_pretrained(
            "black-forest-labs/FLUX.2-klein-base-9B",
            torch_dtype=torch.bfloat16,
            quantization_config=quant_config,
            device_map="cuda",
        )
    except Exception as e:
        print(f"Could not load FLUX.2-klein-base-9B pipeline, skipping FLUX.2 edits: {e}")


    # Read CSV and generate edited images
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, start=1):
            image_url = row.get("image_url")
            prompt = row.get("generated_prompt")

            if not image_url or not prompt:
                continue

            fsn = row.get("fsn") or f"row_{idx}"

            # Prefer local downloaded image if available, fall back to URL
            local_image_path = os.path.join(INPUT_DIR, f"{fsn}.png")
            input_image = None

            if os.path.exists(local_image_path):
                try:
                    input_image = Image.open(local_image_path).convert("RGB")
                except Exception as e:
                    print(f"Failed to load local image for row {idx} (path={local_image_path}): {e}")

            if input_image is None:
                try:
                    input_image = load_image(image_url)
                    print(
                        f"Warning: using remote URL for row {idx} because local file was missing or invalid. "
                        f"Consider running download_images.py first."
                    )
                except Exception as e:
                    print(f"Failed to load image for row {idx} (url={image_url}): {e}")
                    continue

            # # FireRed edit (existing behavior)
            # try:
            #     print(f"Editing with FireRed pipeline for row {idx}...")
            #     fire_image = fire_pipe(image=input_image, prompt=prompt).images[0]
            #     fire_output_path = os.path.join(OUTPUT_DIR, f"{fsn}_edited.png")
            #     fire_image.save(fire_output_path)
            #     print(f"Saved FireRed edited image to {fire_output_path}")
            # except Exception as e:
            #     print(f"FireRed edit failed for row {idx}: {e}")

            # # LongCat-Image edit
            # if longcat_pipe is not None:
            #     try:
            #         print(f"Editing with LongCat-Image-Edit pipeline for row {idx}...")
            #         longcat_image = longcat_pipe(
            #             image=input_image,
            #             prompt=prompt
            #         ).images[0]
            #         longcat_output_path = os.path.join(OUTPUT_DIR, f"{fsn}_longcat_edited.png")
            #         longcat_image.save(longcat_output_path)
            #         print(f"Saved LongCat edited image to {longcat_output_path}")
            #     except Exception as e:
            #         print(f"LongCat edit failed for row {idx}: {e}")

            # Qwen-Image-Edit
            if qwen_pipe is not None:
                try:
                    print(f"Editing with Qwen-Image-Edit-2511 pipeline for row {idx}...")
                    qwen_image = qwen_pipe(
                        image=input_image,
                        prompt=prompt
                    ).images[0]
                    qwen_output_path = os.path.join(OUTPUT_DIR, f"{fsn}_qwen_edited.png")
                    qwen_image.save(qwen_output_path)
                    print(f"Saved Qwen edited image to {qwen_output_path}")
                except Exception as e:
                    print(f"Qwen edit failed for row {idx}: {e}")

            # FLUX.2-klein-base-9B
            if flux_pipe is not None:
                try:
                    print(f"Editing with FLUX.2-klein-base-9B pipeline for row {idx}...")
                    flux_image = flux_pipe(
                        image=input_image,
                        prompt=prompt
                    ).images[0]
                    flux_output_path = os.path.join(OUTPUT_DIR, f"{fsn}_flux_edited.png")
                    flux_image.save(flux_output_path)
                    print(f"Saved FLUX.2 edited image to {flux_output_path}")
                except Exception as e:
                    print(f"FLUX.2 edit failed for row {idx}: {e}")


if __name__ == "__main__":
    main()

# import os
# import csv

# import torch
# from diffusers import DiffusionPipeline
# # from diffusers.quantizers import PipelineQuantizationConfig
# from diffusers.utils import load_image


# OUTPUT_DIR = "generated_images"
# CSV_PATH = "product_interaction_prompts.csv"


# os.makedirs(OUTPUT_DIR, exist_ok=True)

# # # Define the pipeline-specific quantization configuration
# # quant_config = PipelineQuantizationConfig(
# #     quant_backend="bitsandbytes_8bit",
# #     quant_kwargs={"load_in_8bit": True},
# # )

# # Load the pipeline
# pipe = DiffusionPipeline.from_pretrained(
#     "FireRedTeam/FireRed-Image-Edit-1.0",
#     torch_dtype=torch.bfloat16,
#     # quantization_config=quant_config,
#     device_map="balanced",
# )

# # Read CSV and generate edited images
# with open(CSV_PATH, newline="", encoding="utf-8") as f:
#     reader = csv.DictReader(f)
#     for idx, row in enumerate(reader, start=1):
#         image_url = row.get("image_url")
#         prompt = row.get("generated_prompt")

#         if not image_url or not prompt:
#             continue

#         try:
#             input_image = load_image(image_url)
#         except Exception as e:
#             print(f"Failed to load image for row {idx} (url={image_url}): {e}")
#             continue

#         image = pipe(image=input_image, prompt=prompt).images[0]

#         fsn = row.get("fsn") or f"row_{idx}"
#         output_path = os.path.join(OUTPUT_DIR, f"{fsn}_edited.png")

#         image.save(output_path)
#         print(f"Saved edited image to {output_path}")