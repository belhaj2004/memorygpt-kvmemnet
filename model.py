import re
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from unidecode import unidecode

UNK = "unk"


def tokenize(text: str) -> List[str]:
    text = re.sub(r"[^a-zA-Z0-9]", " ", unidecode(text))
    return text.lower().split()


class Vocab:
    def __init__(self, name: str = "vocab"):
        self.name = name
        self._word2index = {}
        self._word2count = {}
        self._index2word = {}
        self._n_words = 0

    def num_words(self) -> int:
        return self._n_words

    def word2index(self, word: str) -> int:
        return self._word2index[word]

    def add_word(self, word: str) -> None:
        if word not in self._word2index:
            self._word2index[word] = self._n_words
            self._word2count[word] = 1
            self._index2word[self._n_words] = word
            self._n_words += 1
        else:
            self._word2count[word] += 1


class KVMemNet(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 128):
        super().__init__()
        self.A = nn.Linear(vocab_size, embed_dim, bias=False)
        self.B = nn.Linear(vocab_size, embed_dim, bias=False)

    def forward(self, question, keys, values):
        q = self.A(question)
        k = self.A(keys)
        v = self.B(values)
        attention = F.softmax(torch.inner(q, k), dim=-1)
        output = attention @ v
        return output, attention


def safe_multihot(text: str, vocab: Vocab) -> np.ndarray:
    unk_index = vocab._word2index.get(UNK)
    indices = []
    for token in tokenize(text):
        idx = vocab._word2index.get(token, unk_index)
        if idx is not None:
            indices.append(idx)
    vector = np.zeros(vocab.num_words(), dtype=np.float32)
    if indices:
        vector[np.unique(indices)] = 1.0
    return vector


def extract_person_name(question: str, database: Dict[str, Dict[str, str]]) -> str | None:
    lowered = question.lower()
    matches = [name for name in database if name in lowered]
    return max(matches, key=len) if matches else None


def trained_answer(question: str, database, vocab, model, device="cpu"):
    person = extract_person_name(question, database)
    if person is None:
        return None
    facts = database[person]
    keys = list(facts.keys())
    values = list(facts.values())
    q = torch.tensor(safe_multihot(question, vocab), device=device)
    k = torch.tensor(np.stack([safe_multihot(x, vocab) for x in keys]), device=device)
    v = torch.tensor(np.stack([safe_multihot(x, vocab) for x in values]), device=device)
    with torch.no_grad():
        output, attention = model(q, k, v)
        logits = torch.inner(model.B(v), output)
        probabilities = torch.softmax(logits, dim=-1)
        index = int(torch.argmax(probabilities).item())
    return {
        "person": person,
        "answer": values[index],
        "relation": keys[index],
        "confidence": float(probabilities[index].item()),
        "rows": [(keys[i], values[i], float(attention[i].item())) for i in range(len(keys))],
    }


RELATION_HINTS = {
    "birth_date": ["when", "born", "birth", "birthday"],
    "birth_place": ["where", "born", "birthplace"],
    "death_date": ["when", "die", "died", "death"],
    "spouse": ["spouse", "marry", "married", "wife", "husband"],
    "party": ["party", "political"],
    "office": ["office", "position", "serve", "held"],
    "alma_mater": ["school", "college", "university", "study", "alma"],
    "occupation": ["occupation", "work", "job", "profession"],
}


def demo_answer(question: str, database: Dict[str, Dict[str, str]]):
    person = extract_person_name(question, database)
    if person is None:
        return None
    tokens = set(tokenize(question))
    facts = database[person]
    scores: List[Tuple[str, float]] = []
    for relation in facts:
        hints = RELATION_HINTS.get(relation, tokenize(relation))
        score = sum(1.0 for hint in hints if hint in tokens)
        if relation == "birth_place" and "where" in tokens:
            score += 1.5
        if relation == "birth_date" and "when" in tokens:
            score += 1.5
        scores.append((relation, score))
    raw = np.array([score for _, score in scores], dtype=np.float32)
    probs = np.exp(raw - raw.max())
    probs = probs / probs.sum()
    index = int(np.argmax(probs))
    relation = scores[index][0]
    return {
        "person": person,
        "answer": facts[relation],
        "relation": relation,
        "confidence": float(probs[index]),
        "rows": [(rel, facts[rel], float(probs[i])) for i, (rel, _) in enumerate(scores)],
    }
