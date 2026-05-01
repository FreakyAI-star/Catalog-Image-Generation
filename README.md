# Multimodal Guided Image Editing for E-Commerce Catalog Enrichment

This repository contains the codebase and report for the **Multimodal Guided Image Editing for E-Commerce Catalog Enrichment** project. 

## Overview
The e-commerce industry faces a critical challenge: generating diverse, high-quality product imagery at scale while maintaining strict fidelity to product identity. Current generative AI tools often corrupt fine-grained details such as logos, text, and product geometry.

This project introduces a comprehensive pipeline for **product-preserving scene enrichment**. Given a clean studio-style product photo and a natural-language instruction, the system generates enriched lifestyle or interaction environments. The methodology emphasizes precise control—ensuring that product identity, text/logos, and material textures remain unaltered while introducing accurate real-world affordances (e.g., natural hand-object interactions).

For a deep dive into the methodology, experimental setup, failure modes, and metrics, please refer to the [`DLCV_project_report.pdf`](DLCV_project_report.pdf) included in this repository.

## Pipeline Architecture
The project follows a three-stage design:
1. **Data Curation & Preparation (`scripts/build_interaction_pairs.py`)**: Automates the formulation of training data and evaluation prompts using VLMs to pair catalog configurations with detailed text scenes.
2. **Image Editing (`run_edit.py`)**: Leverages guided diffusion models (such as FireRed-Image-Edit, LongCat, and Nano Banana Pro) to blend heterogeneous conditions (reference images, text instructions).
3. **Automated Evaluation (`eval_edit.py`)**: Uses a council-of-experts evaluation approach powered by `EditScore` and VLMs (e.g., `Qwen3-VL-8B-Instruct`) to quantitatively score edits across four dimensions: 
   - **Product Identity**
   - **Prompt Following**
   - **Perceptual Realism**
   - **Affordance Plausibility**

## Repository Structure

- `run_edit.py` — The core inference script. Takes items from `product_interaction_prompts.csv`, downloads the raw images, and batches requests to the chosen editing diffusion pipeline.
- `eval_edit.py` — The automated VLM-based scoring script to critique and score generated outputs against unmodified inputs based on the project's rubric.
- `scripts/build_interaction_pairs.py` — Toolkit handling dataset processing, catalog deduplication and the generation of structured interaction pairs.
- `product_interaction_prompts.csv` — Contains evaluation subsets covering varied categories like mobile accessories, appliances, laptops, and small electronics along with complex scene prompts.
- `DLCV_project_report.pdf` — The primary research report detailing experiments, results, architecture choices, and references.
- `requirements.txt` — Project's Python dependencies.

## Setup and Installation

### Prerequisites
* Linux-based OS recommended
* Python >= 3.10
* NVIDIA GPU with sufficient VRAM for diffusion models and VLMs

### Steps
1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd FireImageEdit
   ```

2. **Set up the Conda environment:**
   ```bash
   conda create -n fire_edit python=3.10 -y
   conda activate fire_edit
   ```

3. **Install exact dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *(Note: Based on your hardware, you may need to install platform-optimized binaries for PyTorch from the official site.)*

## Usage

### 1. Generating Image Edits
The underlying pipeline runs inference over the curated dataset.
```bash
python run_edit.py
```
This script sequentially reads the target prompts, downloads images to `downloaded_images/`, applies the diffusion-based editing strategy, and saves results in `generated_images/`.

### 2. Evaluating Image Edits
Once your images are generated, assess their quality on the four-dimensional rubric framework using:
```bash
python eval_edit.py
```
This expects `generated_images/` and outputs evaluation metrics showing identity retention and scene plausibility score.

## Evaluation Baselines
As detailed in the report, several robust baselines were validated using this framework with differing levels of product identity preservation vs scene realism:
* **FireRed-Image-Edit**
* **LongCat-Image-Edit**
* **Nano Banana Pro**

The `eval_edit.py` loop exposes critical patterns regarding how models handle human-object interactions (affordance), screen/text drift, and spatial physics compared to bare visual realism. 

## Author
* **Rohit Doriya** (rohitdoriya@iisc.ac.in)
