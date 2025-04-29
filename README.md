# 🤖 AI Toolkit: Dataset Creator & Fine-Tuning in Colab

Welcome to **AI Toolkit** – a powerful duo of tools for creating high-quality datasets using AI and fine-tuning models seamlessly in Google Colab! Whether you're training LLMs or building custom AI applications, this repo is here to simplify your workflow.

---

# 🤖 AI Toolkit: Dataset Creator & Fine-Tuner in Colab

Welcome to **AI Toolkit** – a powerful duo of tools for creating high-quality datasets using AI and fine-tuning models seamlessly in Google Colab! Whether you're training LLMs or building custom AI applications, this repo is here to simplify your workflow.

---

## 📁 Repository Structure

### 1. `dataset_creator.py`
A smart Python tool for generating AI-ready datasets with minimal effort.

**Features:**
- Utilizes the `ollama` API to generate synthetic data.
- Regex-based cleaning for high-quality output.
- Saves to JSON for ease of use.
- Includes progress tracking with `tqdm`.

**Libraries Used:**
```python
ollama
re
json
tqdm 
random
torch
os
pathlib
zipfile
shutil
transformers
datasets
peft
```

### 2. `colab_fine_tune.py`
A Python script intended to be copied and run directly in Google Colab for model fine-tuning.

**Highlights:**
- Requires user to upload:
  - A model in `.zip` format
  - A dataset in `.jsonl` format
  - Both files should be uploaded to Google Drive
- User defines paths to files
- Code is pasted into a Colab cell and executed step-by-step
- Designed for ease of use with Colab GPU support

---

## 🚀 Getting Started

### Prerequisites
Make sure you have the following installed if you're running locally:
- Python 3.8+
- Required libraries (install with `pip install -r requirements.txt` or manually)
- Google Drive access via Colab (for fine-tuning)

---

## 🛠️ How to Use

### 🔹 Dataset Creator
```bash
python dataset_creator.py
```
- Customize prompts and dataset size inside the script.
- Output is saved as a JSON file.

### 🔹 Colab Fine-Tuning
1. Upload your `.zip` model file and `.jsonl` dataset file to Google Drive.
2. Define the paths to the files in Colab.
3. Copy and paste the contents of `colab_fine_tune.py` into a Colab notebook cell.
4. Run the cell.

---

## 📌 Notes
- The dataset creator is optimized for text-based datasets (e.g., Q&A, summarization, dialogues).
- Make sure to monitor output quality and clean further if needed.
- Fine-tuning should be tested on subsets first to validate configurations.

---

## 📫 Contributions & Feedback
Pull requests and issues are welcome! Let’s build better tools together. 💬

---

Happy fine-tuning! 🧠✨



