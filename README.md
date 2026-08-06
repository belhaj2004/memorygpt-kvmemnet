# MemoryGPT — Interactive Key-Value Memory Network

MemoryGPT turns a coursework notebook into an interactive portfolio project. Users ask natural-language questions about people, receive an answer from a structured memory store, and inspect the relation-level attention behind the result.

## Highlights

- PyTorch Key-Value Memory Network implemented from scratch
- Natural-language biography questions
- Entity matching and memory retrieval
- Attention visualization and memory inspection
- Automatic demo mode for easy public deployment
- Automatic trained-model mode when saved artifacts are present

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Deploy with Streamlit Community Cloud

1. Create a GitHub repository and push these files.
2. Open Streamlit Community Cloud and select **New app**.
3. Choose the repository, branch, and `app.py`.
4. Deploy and copy the public URL into your GitHub About section and LinkedIn post.

The app works immediately in demo mode. For the original trained model, add the three files described in `artifacts/README.md`. Since the full Wikipedia-derived database and model files may be large, Git LFS or a hosted release/download step is preferable.

## Architecture

1. Convert a question, relation keys, and values into multi-hot vectors.
2. Embed questions and keys through matrix **A**.
3. Compute attention over memory keys.
4. Combine embedded values from matrix **B** using the attention distribution.
5. Score candidate values and return the highest-scoring text answer.

## Example questions

- `When was alexander hamilton born?`
- `Who was alexander hamilton's spouse?`
- `What office did margaret thatcher hold?`
- `Where was barack obama born?`

## Repository structure

```text
memorygpt-kvmemnet/
├── app.py
├── model.py
├── requirements.txt
├── artifacts/
├── notebooks/
└── .streamlit/
```

## Limitations

Entity extraction uses exact longest-name matching. The original model was trained on eight biography relations, so paraphrases and unknown people may fail. Demo mode is explicitly a lightweight public preview; trained-model mode uses the neural network from the notebook.
