from pathlib import Path
import pickle

import pandas as pd
import streamlit as st
import torch

from model import KVMemNet, demo_answer, trained_answer

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
    "Birth date": "When was Alexander Hamilton born?",
    "Occupation": "What was Margaret Thatcher's occupation?",
    "Political party": "Which political party was Barack Obama associated with?",
    "Spouse": "Who was Alexander Hamilton's spouse?",
    "Office held": "What office did Margaret Thatcher hold?",
    "Education": "Where did Barack Obama study?",
}

ARTIFACTS = Path("artifacts")


@st.cache_resource
def load_runtime():
    model_path = ARTIFACTS / "kvmemnet_model_final.pt"
    db_path = ARTIFACTS / "data.pkl"
    vocab_path = ARTIFACTS / "vocab.pkl"
    if not all(path.exists() for path in (model_path, db_path, vocab_path)):
        return "demo", DEMO_DB, None, None

    with db_path.open("rb") as handle:
        database = pickle.load(handle)
    with vocab_path.open("rb") as handle:
        vocab = pickle.load(handle)

    relations = {
        "office",
        "birth_date",
        "birth_place",
        "party",
        "death_date",
        "spouse",
        "alma_mater",
        "occupation",
    }
    database = {
        name: {key: value for key, value in facts.items() if key in relations}
        for name, facts in database.items()
        if any(key in relations for key in facts)
    }

    model = KVMemNet(vocab.num_words(), 128)
    state = torch.load(model_path, map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.eval()
    return "trained", database, vocab, model


def choose_example(question: str) -> None:
    st.session_state.question = question
    st.session_state.run_search = True


mode, database, vocab, model = load_runtime()

if "question" not in st.session_state:
    st.session_state.question = "Where was Barack Obama born?"
if "run_search" not in st.session_state:
    st.session_state.run_search = False

st.title("🧠 MemoryGPT")
st.caption(
    "Ask a factual question about a person in the model's memory and see which memory slot influenced the answer."
)

with st.sidebar:
    st.subheader("Runtime")
    st.success("Trained model loaded" if mode == "trained" else "Portfolio demo mode")
    st.write(f"People available: **{len(database):,}**")

    st.divider()
    st.subheader("People available")
    for person in sorted(database):
        st.write(f"• {person.title()}")

st.info(
    "This is a focused biography question-answering demo, not a general chatbot. "
    "Ask about a person shown in the sidebar using one of the supported fact types below."
)

st.markdown("### What can I ask?")
st.write(
    "Choose an example or write a similar question using a person available in the model's memory."
)

labels = list(SUPPORTED_QUESTIONS)
for row_start in range(0, len(labels), 4):
    row_labels = labels[row_start : row_start + 4]
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
- **Spouse:** Who was `[person]`'s spouse?
- **Office held:** What office did `[person]` hold?
- **Education:** Where did `[person]` study?

The trained model may also support **death date** questions when that fact exists in memory.
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
        "Search memory", type="primary", use_container_width=True
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
                "I could not identify a person stored in memory. Choose a person from the sidebar "
                "and use one of the supported question patterns."
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
                st.info(f"Matched entity: {result['person'].title()}")

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
                st.markdown(
                    "The question is converted into a multi-hot vector. The network embeds the question and memory keys, "
                    "computes attention across relations, combines the value embeddings, and selects the highest-scoring value."
                    if mode == "trained"
                    else "This public preview uses relation-keyword scoring so the repository runs without large private training artifacts. "
                    "Add your saved model, database, and vocabulary to activate the original neural-network inference path automatically."
                )

st.divider()
st.caption(
    "Built from a PyTorch Key-Value Memory Network trained on structured Wikipedia biography facts."
)
