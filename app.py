from pathlib import Path
import pickle
import urllib.request

import pandas as pd
import streamlit as st
import torch

from model import KVMemNet, demo_answer, trained_answer, Vocab

st.set_page_config(page_title="MemoryGPT", page_icon="🧠", layout="wide")

DEMO_DB = {
    "alexander hamilton": {
        "office": "1st United States Secretary of the Treasury",
        "birth_date": "11 January 1755 or 1757",
        "birth_place": "Charlestown, Nevis",
        "party": "Federalist Party",
        "death_date": "12 July 1804",
        "spouse": "Elizabeth Schuyler Hamilton",
        "alma_mater": "King's College (now Columbia University)",
        "occupation": "Statesman, lawyer, military commander, and economist",
    },
    "barack obama": {
        "office": "44th President of the United States",
        "birth_date": "4 August 1961",
        "birth_place": "Honolulu, Hawaii",
        "party": "Democratic Party",
        "spouse": "Michelle Obama",
        "alma_mater": "Columbia University and Harvard Law School",
        "occupation": "Politician, lawyer, and author",
    },
    "margaret thatcher": {
        "office": "Prime Minister of the United Kingdom",
        "birth_date": "13 October 1925",
        "birth_place": "Grantham, Lincolnshire, England",
        "party": "Conservative Party",
        "death_date": "8 April 2013",
        "spouse": "Denis Thatcher",
        "alma_mater": "Somerville College, Oxford",
        "occupation": "Politician and barrister",
    },
}

SUPPORTED_QUESTIONS = {
    "Birth place": "Where was Barack Obama born?",
    "Birth date": "When was Barack Obama born?",
    "Occupation": "What was Barack Obama's occupation?",
    "Political party": "Which political party was Barack Obama associated with?",
    "Spouse": "Who was Barack Obama married to?",
    "Office held": "What office did Barack Obama hold?",
    "Education": "Where did Barack Obama study?",
}

ARTIFACTS = Path("artifacts")
MODEL_PATH = ARTIFACTS / "kvmemnet_model_final.pt"
DB_PATH = ARTIFACTS / "data.pkl"
VOCAB_PATH = ARTIFACTS / "vocab.pkl"

MODEL_URL = (
    "https://huggingface.co/belhaj2004/"
    "memorygpt-kvmemnet/resolve/main/kvmemnet_model_final.pt"
)

RELATIONS = {
    "office",
    "birth_date",
    "birth_place",
    "party",
    "death_date",
    "spouse",
    "alma_mater",
    "occupation",
}


def ensure_model():
    """Download the trained model from Hugging Face if Streamlit does not have it yet."""
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    if not MODEL_PATH.exists():
        with st.spinner("Downloading trained MemoryGPT model from Hugging Face..."):
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

    return MODEL_PATH


@st.cache_resource
def load_runtime():
    # data.pkl and vocab.pkl remain in the GitHub artifacts/ folder.
    if not DB_PATH.exists() or not VOCAB_PATH.exists():
        return "demo", DEMO_DB, None, None, (
            "data.pkl or vocab.pkl is missing from the artifacts folder."
        )

    try:
        model_path = ensure_model()

        with DB_PATH.open("rb") as handle:
            database = pickle.load(handle)

        # Vocab is imported above so pickle can resolve the class created in Colab.
        with VOCAB_PATH.open("rb") as handle:
            vocab = pickle.load(handle)

        database = {
            name: {key: value for key, value in facts.items() if key in RELATIONS}
            for name, facts in database.items()
            if isinstance(facts, dict) and any(key in RELATIONS for key in facts)
        }

        model = KVMemNet(vocab.num_words(), 128)
        state = torch.load(model_path, map_location="cpu")

        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]

        model.load_state_dict(state)
        model.eval()

        return "trained", database, vocab, model, None

    except Exception as exc:
        # Keep the public site alive even if Hugging Face or an artifact temporarily fails.
        return "demo", DEMO_DB, None, None, str(exc)


def choose_example(question):
    st.session_state.question = question
    st.session_state.run_search = True


mode, database, vocab, model, load_error = load_runtime()

if "question" not in st.session_state:
    st.session_state.question = "Where was Barack Obama born?"
if "run_search" not in st.session_state:
    st.session_state.run_search = False

st.title("🧠 MemoryGPT")
st.caption(
    "Ask a factual question about a person in the model's memory and "
    "see which memory slot influenced the answer."
)

with st.sidebar:
    st.subheader("Runtime")

    if mode == "trained":
        st.success("Trained model loaded")
        st.write(f"People in memory: **{len(database):,}**")
        st.caption("Model weights hosted on Hugging Face.")
    else:
        st.warning("Portfolio demo mode")
        st.write(f"People available: **{len(database):,}**")
        if load_error:
            with st.expander("Why isn't the trained model loaded?"):
                st.code(load_error)

    st.divider()
    st.subheader("Find a person")

    person_search = st.text_input(
        "Search the memory",
        placeholder="e.g. Barack Obama",
        label_visibility="collapsed",
    )

    if person_search:
        needle = person_search.lower().strip()
        matches = [
            person for person in database
            if needle in str(person).lower()
        ][:20]

        if matches:
            for person in matches:
                st.write(f"• {str(person).title()}")
        else:
            st.caption("No matching person found.")

st.info(
    "This is a focused biography question-answering demo, not a general chatbot. "
    "Ask about a person stored in the model using one of the supported fact types below."
)

st.markdown("### What can I ask?")
st.write(
    "Choose an example or write a similar question using a person available in the model's memory."
)

labels = list(SUPPORTED_QUESTIONS)

for row_start in range(0, len(labels), 4):
    row_labels = labels[row_start:row_start + 4]
    columns = st.columns(len(row_labels))

    for column, label in zip(columns, row_labels):
        with column:
            st.button(
                label,
                key=f"example_{label}",
                use_container_width=True,
                on_click=choose_example,
                args=(SUPPORTED_QUESTIONS[label],),
            )

with st.expander("See supported question patterns"):
    st.markdown(
        """
- **Birth place:** Where was `[person]` born?
- **Birth date:** When was `[person]` born?
- **Occupation:** What was `[person]`'s occupation?
- **Political party:** Which political party was `[person]` associated with?
- **Spouse:** Who was `[person]` married to?
- **Office held:** What office did `[person]` hold?
- **Education:** Where did `[person]` study?
- **Death date:** When did `[person]` die? *(when that fact exists in memory)*
"""
    )

st.markdown("### Ask MemoryGPT")

with st.form("question_form"):
    question = st.text_input(
        "Question",
        key="question",
        placeholder="Example: Where was Barack Obama born?",
        label_visibility="collapsed",
    )
    submitted = st.form_submit_button(
        "Search memory",
        type="primary",
        use_container_width=True,
    )

should_search = submitted or st.session_state.run_search

if st.session_state.run_search:
    st.session_state.run_search = False

if should_search:
    cleaned_question = question.strip()

    if not cleaned_question:
        st.warning("Enter a question or select one of the examples above.")
    else:
        result = (
            trained_answer(cleaned_question, database, vocab, model)
            if mode == "trained"
            else demo_answer(cleaned_question, database)
        )

        if result is None:
            st.warning(
                "I could not identify a person stored in memory. "
                "Search for a person in the sidebar and use one of the supported question patterns."
            )
        else:
            left, right = st.columns([1.1, 1])

            with left:
                st.subheader("Answer")
                st.markdown(f"### {result['answer']}")
                st.metric(
                    "Selected relation",
                    result["relation"].replace("_", " ").title(),
                )
                st.metric("Model confidence", f"{result['confidence']:.1%}")
                st.info(f"Matched entity: {str(result['person']).title()}")

            with right:
                st.subheader("Attention over memory")
                frame = pd.DataFrame(
                    result["rows"],
                    columns=["Relation", "Stored value", "Attention"],
                )
                frame = frame.sort_values("Attention", ascending=False)
                st.bar_chart(frame.set_index("Relation")["Attention"])

            st.subheader("Memory inspection")
            shown = frame.copy()
            shown["Attention"] = shown["Attention"].map(
                lambda value: f"{value:.2%}"
            )
            st.dataframe(shown, use_container_width=True, hide_index=True)

            with st.expander("How this answer was produced"):
                if mode == "trained":
                    st.markdown(
                        "The question is converted into a multi-hot vector. "
                        "The trained Key-Value Memory Network embeds the question and memory keys, "
                        "computes attention across relations, combines the value embeddings, "
                        "and selects the highest-scoring value."
                    )
                else:
                    st.markdown(
                        "The fallback public preview uses relation-keyword scoring. "
                        "When `data.pkl` and `vocab.pkl` are present, the app downloads the "
                        "trained `.pt` weights from Hugging Face and activates neural-network inference."
                    )

st.divider()
st.caption(
    "Built from a PyTorch Key-Value Memory Network trained on structured Wikipedia biography facts."
)
