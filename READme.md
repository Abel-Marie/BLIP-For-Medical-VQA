
---

````markdown
# 🩺 Fine-Tuning BLIP for Medical Visual Question Answering (VQA)

This repository contains a PyTorch-based notebook for fine-tuning the **Salesforce BLIP** (Bootstrapping Language-Image Pre-training) model for the specific task of **Visual Question Answering on medical imagery**. The model is trained on the **PathVQA** dataset, which includes pathology images and associated question-answer pairs.

The project culminates in an interactive web interface built with **Gradio**, allowing users to upload their own medical images, ask questions, and receive answers from the fine-tuned model.

---

## ✨ Key Features

- **Hugging Face Integration**  
  Uses the `transformers` library to load the pre-trained `Salesforce/blip-vqa-base` model and the `datasets` library to access `flaviagiammarino/path-vqa`.

- **Efficient Training**
  - ✅ Mixed-precision training (`torch.cuda.amp`) for speed and memory efficiency.
  - ✅ Gradient accumulation to simulate a larger batch size on limited hardware.

- **Custom Data Handling**  
  A `torch.utils.data.Dataset` subclass processes images and question-answer pairs on the fly.

- **End-to-End Workflow**  
  Covers the full pipeline: dataset loading, preprocessing, model training, evaluation, and saving.

- **Interactive Demo**  
  Launches a **Gradio** web app for real-time inference and model testing.

---

## 🚀 Getting Started

Follow these steps to run the project on your own machine or in Google Colab.

### 1. Clone the Repository

```bash
git clone https://github.com/Abel-Marie/BLIP-For-Medical-VQA.git
cd BLIP-For-Medical-VQA
````

### 2. Install Dependencies

```bash
pip install transformers[torch] datasets -q gradio
```

### 3. Run the Notebook

Open `Fine-tuning BLIP for Medical VQA.ipynb` in Jupyter or Google Colab.

Make sure to **enable GPU** for faster training.

Run all the cells in order to:

* ✅ Load libraries
* ✅ Configure hyperparameters
* ✅ Load and preprocess the PathVQA dataset
* ✅ Fine-tune the BLIP model
* ✅ Save the trained model
* ✅ Launch the Gradio demo interface

---

## ⚙️ Configuration

All key settings are centralized in the `Config` class for easy tuning.

| Parameter                     | Default                     | Description                            |
| ----------------------------- | --------------------------- | -------------------------------------- |
| `MODEL_NAME`                  | `Salesforce/blip-vqa-base`  | Base BLIP model from Hugging Face      |
| `DATASET_NAME`                | `flaviagiammarino/path-vqa` | Medical VQA dataset                    |
| `OUTPUT_DIR`                  | `blip_vqa_finetuned`        | Directory to save fine-tuned model     |
| `TRAIN_SAMPLES`               | `2000`                      | Number of training examples            |
| `VAL_SAMPLES`                 | `500`                       | Number of validation examples          |
| `EPOCHS`                      | `3`                         | Training epochs                        |
| `BATCH_SIZE`                  | `2`                         | Mini-batch size                        |
| `GRADIENT_ACCUMULATION_STEPS` | `16`                        | Steps to accumulate gradients          |
| `LEARNING_RATE`               | `5e-5`                      | Optimizer learning rate                |
| `MAX_TEXT_LENGTH`             | `32`                        | Max token length for questions/answers |

> 💡 **Note:** The effective batch size = `BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS` (e.g., 2 × 16 = 32). Adjust based on your GPU.

---

## 💾 Model Saving & Inference

After training, the model and processor are saved to the specified `OUTPUT_DIR`. You can reload them later for inference:

```python
from transformers import BlipProcessor, BlipForQuestionAnswering

output_dir = "blip_vqa_finetuned"

processor = BlipProcessor.from_pretrained(output_dir)
model = BlipForQuestionAnswering.from_pretrained(output_dir)
```

---

## 🤝 Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request for:

* 💡 New features
* 🐞 Bug fixes
* 📚 Documentation improvements

---

## 🙏 Acknowledgments

* 🧠 **Salesforce** for releasing the [BLIP model](https://github.com/salesforce/BLIP)
* 📸 **Flavia Giammarino** for curating the [PathVQA dataset](https://huggingface.co/datasets/flaviagiammarino/path-vqa)
* 🧰 **Hugging Face** and **Gradio** for enabling fast and accessible machine learning development

---


