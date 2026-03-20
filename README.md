# Fundus-Fusion-AI

Fundus-Fusion-AI is a project for image stitching and AI-based analysis of fundus images, including glaucoma detection and retinal screening. The project combines image processing and deep learning models for medical imaging tasks.

## Project Structure

- `image Stitching/` - Web app for image stitching and screening
  - `app.py` - Main Flask app
  - `static/` - JS, CSS, and sample images
  - `templates/` - HTML templates
  - `uploads/` - Uploaded images (ignored by git)
  - `outputs/` - Output images (ignored by git)
  - `stitching/` - Image processing and screening modules
- `model/RETFound/` - Deep learning models and scripts
  - `engine_finetune.py`, `main_finetune.py`, `predict.py` - Model training and inference
  - `dataset/` - Data for training, validation, and testing
  - `output_dir/`, `output_logs/` - Model outputs and logs (ignored by git)
  - `weights/` - Pretrained model weights

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/lohith889/Fundus-Fusion-AI.git
   ```
2. Install dependencies:
   ```bash
   pip install -r image Stitching/requirements.txt
   pip install -r model/RETFound/requirements.txt
   ```
3. Run the web app:
   ```bash
   cd "image Stitching"
   python app.py
   ```

## Notes
- Uploaded images, outputs, model weights, and logs are ignored by git for privacy and size reasons.
- For model training and inference, see scripts in `model/RETFound/`.

## License
See `model/RETFound/LICENSE` for license information.
