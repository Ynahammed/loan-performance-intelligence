"""
Retrieval over the data dictionary and the validation rules.

The corpus is two small files, so this is TF-IDF over sentence-level
chunks rather than embeddings and a vector store. That is a deliberate
choice, not a shortcut: the whole corpus is a few kilobytes, an embedding
model would add a dependency and a download to a system that currently
needs neither, and retrieval quality on a field-definition lookup is
dominated by exact term matching -- "what is ltv_band" wants the row that
literally says `ltv_band`.

What this buys is the thing that matters for the copilot: when a reviewer
asks what a field means, the answer is quoted from the organizer's own
documentation with the source named, rather than recalled from a model's
training data. A wrong-but-fluent definition of a mortgage field is
exactly the kind of error nobody catches.

PHASE: 10
STATUS: implemented.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import DATA_DIR

logger = logging.getLogger(__name__)


@dataclass
class Passage:
    text: str
    source: str
    identifier: str = ""
    key: str = ""

    def key_tokens(self) -> set:
        """Tokens of the field name or rule id this passage defines."""
        return {t for t in re.split(r"[^a-z0-9]+", (self.key or "").lower()) if t}

    def cite(self) -> str:
        if self.identifier:
            return "{} ({})".format(self.source, self.identifier)
        return self.source


def load_corpus(data_dir: Path | str = None) -> list:
    """Chunk the data dictionary and validation rules into passages.

    Markdown table rows are chunked individually: one row is one field
    definition, which is exactly the retrieval unit a field lookup wants.
    """
    data_dir = Path(data_dir or DATA_DIR)
    passages = []

    dictionary = data_dir / "data_dictionary.md"
    if dictionary.exists():
        current_section = "data_dictionary.md"
        for line in dictionary.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                current_section = line.lstrip("#").strip()
                continue
            if line.startswith("|") and line.count("|") >= 3:
                cells = [c.strip() for c in line.strip("|").split("|")]
                if len(cells) < 2 or set(cells[0]) <= set("-: "):
                    continue
                if cells[0].lower() in ("field", "column", "name"):
                    continue
                passages.append(Passage(
                    text="{}: {}".format(cells[0], cells[1]),
                    source="data_dictionary.md",
                    identifier=current_section,
                    key=cells[0],
                ))
            elif len(line) > 40 and not line.startswith(">"):
                passages.append(Passage(
                    text=line, source="data_dictionary.md",
                    identifier=current_section,
                ))

    rules_path = data_dir / "validation_rules.json"
    if rules_path.exists():
        payload = json.loads(rules_path.read_text(encoding="utf-8"))
        for rule in payload.get("rules", []):
            passages.append(Passage(
                text="{}: {} (severity: {})".format(
                    rule.get("rule_id"), rule.get("description"),
                    rule.get("severity")),
                source="validation_rules.json",
                identifier=rule.get("rule_id", ""),
                key=rule.get("rule_id", ""),
            ))

    logger.info("Loaded %d passages for retrieval", len(passages))
    return passages


class DocumentationRetriever:
    """TF-IDF retrieval with an explicit relevance floor.

    The floor matters more than the ranking. Without it, a question the
    corpus cannot answer still returns its three least-irrelevant
    passages, the model dutifully answers from them, and the result is a
    confident answer built on unrelated text. Below the floor this returns
    nothing and the caller says so.
    """

    def __init__(self, passages: list = None, min_score: float = 0.12):
        self.passages = passages if passages is not None else load_corpus()
        self.min_score = min_score
        if not self.passages:
            self._vectorizer = None
            self._matrix = None
            return
        self._vectorizer = TfidfVectorizer(
            lowercase=True,
            token_pattern=r"[A-Za-z_][A-Za-z0-9_]+",
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        self._matrix = self._vectorizer.fit_transform(
            [p.text for p in self.passages]
        )

    def search(self, question: str, top_k: int = 3) -> list:
        """Passages that NAME the thing being asked about, best first.

        A cosine floor alone does not work here, and the numbers say so.
        Measured on this corpus, "what is the borrower's favourite
        colour?" -- a question the documentation cannot answer -- scores
        0.198 against `credit_score_band`, because that row reads
        "Borrower credit score band". Two legitimate lookups score LOWER
        (0.179 and 0.183). No threshold separates them. IDF does not help
        either: the corpus has 496 terms and "borrower" carries the
        maximum IDF of 4.30, so rarity and relevance have come apart.

        So retrieval additionally requires the question to share a token
        with the passage's FIELD NAME, not merely with its prose. That is
        the correct semantics for a field lookup -- a question about
        `ltv_band` names it, one about favourite colours does not -- and
        it is a hard gate rather than a tuned number.

        The limitation this accepts, stated rather than hidden: a question
        phrased entirely without the field name ("how is lateness
        defined?") retrieves nothing and the caller reports that the
        documentation does not cover it. For a field-definition lookup
        that is the right failure -- silence beats a confident answer
        assembled from unrelated rows.
        """
        if self._vectorizer is None or not (question or "").strip():
            return []
        normalised = re.sub(r"[^A-Za-z0-9_]+", " ", question).strip()
        query_tokens = {
            t for t in re.split(r"[^a-z0-9]+", normalised.lower()) if t
        }

        query = self._vectorizer.transform([normalised])
        scores = (self._matrix @ query.T).toarray().ravel()
        order = np.argsort(-scores)[:max(top_k * 5, 10)]

        hits = []
        for i in order:
            if scores[i] < self.min_score:
                continue
            passage = self.passages[i]
            key_tokens = passage.key_tokens()
            if key_tokens and not (key_tokens & query_tokens):
                continue
            hits.append({"passage": passage, "score": float(scores[i])})
            if len(hits) >= top_k:
                break
        return hits

    def context_block(self, question: str, top_k: int = 3) -> tuple:
        """Retrieved text formatted for a prompt, plus the citations.

        Returns ("", []) when nothing clears the floor, so the caller can
        tell the difference between "here is the answer" and "the
        documentation does not cover this".
        """
        hits = self.search(question, top_k)
        if not hits:
            return "", []
        lines, citations = [], []
        for hit in hits:
            passage = hit["passage"]
            lines.append("- {} [source: {}]".format(passage.text, passage.cite()))
            citations.append(passage.cite())
        return "\n".join(lines), citations


def answer_from_documentation(
    question: str, retriever: DocumentationRetriever = None
) -> dict:
    """Look up a field or rule. Returns the grounded answer and its sources.

    The deterministic path quotes the retrieved passage verbatim, which is
    the correct behaviour for a definition lookup: there is nothing to
    paraphrase and paraphrasing is where errors enter.
    """
    retriever = retriever or DocumentationRetriever()
    block, citations = retriever.context_block(question)
    if not block:
        return {
            "question": question,
            "answer": "The provided documentation does not cover this. No "
                      "definition was found in data_dictionary.md or "
                      "validation_rules.json.",
            "citations": [],
            "grounded": False,
        }
    hits = retriever.search(question)
    return {
        "question": question,
        "answer": hits[0]["passage"].text,
        "supporting": [h["passage"].text for h in hits[1:]],
        "citations": citations,
        "grounded": True,
        "retrieved_block": block,
    }
