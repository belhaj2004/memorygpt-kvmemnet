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
    relations = {"office", "birth_date", "birth_place", "party", "death_date", "spouse", "alma_mater", "occupation"}
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

mode, database, vocab, model = load_runtime()

st.title("🧠 MemoryGPT")
st.caption("An interactive Key-Value Memory Network that reveals which memory slot influenced each answer.")

with st.sidebar:
    st.subheader("Runtime")
    st.success("Trained model loaded" if mode == "trained" else "Portfolio demo mode")
    st.write(f"People available: **{len(database):,}**")
    st.markdown("**Try asking**")
    examples = [
        "When was alexander hamilton born?",
        "Who was alexander hamilton's spouse?",
        "What office did margaret thatcher hold?",
        "Where was barack obama born?",
    ]
    for example in examples:
        st.code(example, language=None)

question = st.text_input("Ask about a person in memory", value="When was alexander hamilton born?")
ask = st.button("Search memory", type="primary", use_container_width=True)

if ask or question:
    result = trained_answer(question, database, vocab, model) if mode == "trained" else demo_answer(question, database)
    if result is None:
        st.warning("I could not identify a person stored in memory. Try one of the examples in the sidebar.")
    else:
        left, right = st.columns([1.1, 1])
        with left:
            st.subheader("Answer")
            st.markdown(f"### {result['answer']}")
            st.metric("Selected relation", result["relation"].replace("_", " ").title())
            st.metric("Model confidence", f"{result['confidence']:.1%}")
            st.info(f"Matched entity: {result['person'].title()}")
        with right:
            st.subheader("Attention over memory")
            frame = pd.DataFrame(result["rows"], columns=["Relation", "Stored value", "Attention"])
            frame = frame.sort_values("Attention", ascending=False)
            st.bar_chart(frame.set_index("Relation")["Attention"])

        st.subheader("Memory inspection")
        shown = frame.copy()
        shown["Attention"] = shown["Attention"].map(lambda value: f"{value:.2%}")
        st.dataframe(shown, use_container_width=True, hide_index=True)

        with st.expander("How this answer was produced"):
            st.markdown(
                "The question is converted into a multi-hot vector. The network embeds the question and memory keys, "
                "computes attention across relations, combines the value embeddings, and selects the highest-scoring value."
                if mode == "trained" else
                "This public preview uses relation-keyword scoring so the repository runs without large private training artifacts. "
                "Add your saved model, database, and vocabulary to activate the original neural-network inference path automatically."
            )

st.divider()
st.caption("Built from a PyTorch Key-Value Memory Network trained on structured Wikipedia biography facts.")
