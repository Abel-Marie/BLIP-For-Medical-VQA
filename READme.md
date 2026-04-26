# Fine-Tuning BLIP for Medical Visual Question Answering (VQA)

> Fine-tunes the **Salesforce BLIP** vision-language model on the **PathVQA** medical-imaging dataset for visual question answering on pathology images. Includes a Gradio demo for interactive inference.

## What this does

- Loads the pre-trained `Salesforce/blip-vqa-base` model via Hugging Face `transformers`.
- Loads the `flaviagiammarino/path-vqa` medical pathology VQA dataset via `datasets`.
- Fine-tunes the model with **mixed-precision (`torch.cuda.amp`)** and **gradient accumulation** to fit on a single Colab/Kaggle GPU.
- Saves the fine-tuned weights and processor.
- Launches a **Gradio** web app so you can upload a pathology image, ask a question, and get the model's answer in real time.

## Files

| File | What |
|---|---|
| `notebook/fine_tuning_BLIP_for_medical_VQA.ipynb` | End-to-end training notebook: data loading → training → eval → save → Gradio demo. |
| `requirements.txt` | Python dependencies. |
| `READme.md` | This file. |
| `LICENSE` | MIT. |

## Install

```bash
git clone https://github.com/Abel-Marie/BLIP-For-Medical-VQA.git
cd BLIP-For-Medical-VQA
pip install -r requirements.txt
```

## How to run

Open `notebook/fine_tuning_BLIP_for_medical_VQA.ipynb` in Jupyter / Colab / Kaggle. Enable GPU. Run cells top-to-bottom:

1. Install dependencies and load libraries.
2. Set hyperparameters via the `Config` class.
3. Load and preprocess the PathVQA dataset.
4. Fine-tune the BLIP model.
5. Save the trained model and processor.
6. Launch the Gradio demo.

## Configuration

All hyperparameters live in the `Config` class in the notebook:

| Parameter | Default | Description |
|---|---|---|
| `MODEL_NAME` | `Salesforce/blip-vqa-base` | Base BLIP model from Hugging Face. |
| `DATASET_NAME` | `flaviagiammarino/path-vqa` | Medical VQA dataset on HF. |
| `OUTPUT_DIR` | `blip_vqa_finetuned` | Where to save the fine-tuned model. |
| `TRAIN_SAMPLES` | `2000` | Number of training examples. |
| `VAL_SAMPLES` | `500` | Number of validation examples. |
| `EPOCHS` | `3` | Training epochs. |
| `BATCH_SIZE` | `2` | Mini-batch size (per step). |
| `GRADIENT_ACCUMULATION_STEPS` | `16` | Effective batch size = `BATCH_SIZE × GRADIENT_ACCUMULATION_STEPS` = 32. |
| `LEARNING_RATE` | `5e-5` | AdamW learning rate. |
| `MAX_TEXT_LENGTH` | `32` | Max token length for questions/answers. |

## Reload for inference

```python
from transformers import BlipProcessor, BlipForQuestionAnswering

output_dir = "blip_vqa_finetuned"
processor = BlipProcessor.from_pretrained(output_dir)
model = BlipForQuestionAnswering.from_pretrained(output_dir)
```

## Sample outputs

> _After fine-tuning, paste 2–3 (image, question, predicted answer) examples here as a screenshot or table._

## Acknowledgments

- **Salesforce** — [BLIP](https://github.com/salesforce/BLIP).
- **Flavia Giammarino** — [PathVQA dataset](https://huggingface.co/datasets/flaviagiammarino/path-vqa).
- **Hugging Face** + **Gradio** — for the rest of the stack.

## License

[MIT](LICENSE) © 2026 Abel Marie Shiferaw
