# 🧠 MemoryGPT

MemoryGPT is an interactive **Key-Value Memory Network** demo built with **PyTorch** and **Streamlit**. It answers factual biography questions by retrieving information from structured memory and visualizing the model's attention over available relations.

## Features

- Search from a large collection of people stored in memory
- Select a person and generate supported questions automatically
- Ask about birth place, birth date, occupation, political party, spouse, office, education, and death date
- View the model's predicted answer and confidence
- Inspect attention scores and memory values
- Trained model weights are hosted on Hugging Face

## Tech Stack

- Python
- PyTorch
- Streamlit
- Pandas
- Hugging Face

## Live Demo

👉 https://memorygpt-kvmemnet-adib.streamlit.app/

## How it works

The question and stored memory keys are converted into vector representations. The Key-Value Memory Network compares the question with relevant memory slots, applies attention over the stored relations, and uses the resulting representation to retrieve the most likely answer.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app expects `data.pkl` and `vocab.pkl` inside the `artifacts/` directory. The trained `.pt` model is downloaded automatically from Hugging Face when needed.

---

Built as an interactive demonstration of Key-Value Memory Networks and attention-based retrieval.
