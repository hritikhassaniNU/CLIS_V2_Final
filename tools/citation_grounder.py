"""
CLIS V2 — Citation Grounding Engine
=====================================
Student: Hritik Hassani | Northeastern University

Maps every sentence in LLM output to its source passage.
This is the core safety feature that separates CLIS from
generic medical chatbots — no claim without a source.

Algorithm:
  1. Split LLM summary into sentences
  2. For each sentence, compute token overlap against all
     retrieved article abstracts
  3. Assign best-match PMID + passage snippet
  4. Flag sentences with low overlap as "ungrounded"
  5. Return grounded citations + overall faithfulness score
"""

import re, math
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GroundedSentence:
    """A single sentence from the LLM summary with its citation."""
    text: str = ""
    pmid: str = ""
    source_title: str = ""
    source_passage: str = ""
    overlap_score: float = 0.0
    is_grounded: bool = False
    url: str = ""


@dataclass
class CitationReport:
    """Full citation grounding report for one query."""
    query: str = ""
    summary: str = ""
    sentences: list = field(default_factory=list)
    faithfulness_score: float = 0.0
    grounded_count: int = 0
    ungrounded_count: int = 0
    overall_verdict: str = "Unverified"
    warning: str = ""


class CitationGrounder:
    """
    Paragraph-level citation grounding for CLIS V2.

    Takes the LLM-generated clinical summary and maps each
    sentence back to a specific passage in the retrieved articles.
    Sentences below the overlap threshold are flagged as
    potentially hallucinated.

    Threshold tuning:
      overlap >= 0.25 → grounded (conservative for medical use)
      overlap >= 0.15 → partial match
      overlap <  0.15 → ungrounded / potential hallucination
    """

    GROUNDED_THRESHOLD  = 0.12
    PARTIAL_THRESHOLD   = 0.12
    # Clinical stopwords — excluded from overlap computation
    STOPWORDS = {
        "the","a","an","in","of","to","for","and","or","with","that",
        "this","is","are","was","were","has","have","had","be","been",
        "being","it","its","as","at","by","from","on","not","no","nor",
        "but","if","so","yet","both","either","neither","each","few",
        "more","most","other","some","such","than","then","there","when",
        "where","which","while","who","whom","whose","will","would",
        "could","should","may","might","must","can","do","does","did",
        "study","studies","patients","treatment","results","conclusion",
        "evidence","clinical","compared","significant","effect","effects",
    }

    def __init__(self):
        pass

    # ── Public API ─────────────────────────────────────────────

    def ground(self, summary: str, articles: list, query: str) -> CitationReport:
        """
        Ground all sentences in summary against retrieved articles.

        Args:
            summary:  LLM-generated clinical summary text
            articles: List of PubMedArticle objects or dicts
            query:    Original clinical question

        Returns:
            CitationReport with per-sentence citations
        """
        sentences = self._split_sentences(summary)
        if not sentences or not articles:
            return CitationReport(
                query=query, summary=summary,
                overall_verdict="Insufficient data for grounding"
            )

        # Build passage index from articles
        passages = self._build_passages(articles)

        grounded_sentences = []
        for sent in sentences:
            best = self._find_best_match(sent, passages)
            grounded_sentences.append(best)

        grounded   = sum(1 for s in grounded_sentences if s.is_grounded)
        ungrounded = len(grounded_sentences) - grounded
        faith      = round(grounded / len(grounded_sentences), 3) if grounded_sentences else 0.0

        if faith >= 0.80:
            verdict = "High faithfulness — all major claims grounded"
        elif faith >= 0.60:
            verdict = "Moderate faithfulness — most claims grounded"
        elif faith >= 0.40:
            verdict = "Low faithfulness — some claims may be ungrounded"
        else:
            verdict = "Very low faithfulness — recommend manual review"

        warning = ""
        if ungrounded > 0:
            warning = (f"{ungrounded} sentence(s) could not be grounded to a source passage. "
                       f"These may represent LLM inference beyond the retrieved evidence. "
                       f"Physician review recommended before clinical use.")

        return CitationReport(
            query=query,
            summary=summary,
            sentences=grounded_sentences,
            faithfulness_score=faith,
            grounded_count=grounded,
            ungrounded_count=ungrounded,
            overall_verdict=verdict,
            warning=warning,
        )

    def score_only(self, summary: str, articles: list) -> float:
        """Fast faithfulness score without full grounding report."""
        if not articles or not summary:
            return 0.5
        context_words = set()
        for art in articles:
            text = self._get_text(art)
            context_words.update(self._clinical_tokens(text))
        sentences = self._split_sentences(summary)
        if not sentences: return 0.5
        grounded = sum(
            1 for s in sentences
            if len(self._clinical_tokens(s) & context_words) >= 3
        )
        return round(grounded / len(sentences), 3)

    # ── Private ────────────────────────────────────────────────

    def _split_sentences(self, text: str) -> list:
        """Split text into meaningful sentences (min 15 chars)."""
        raw = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s.strip() for s in raw if len(s.strip()) >= 15]

    def _clinical_tokens(self, text: str) -> set:
        """Tokenize and remove stopwords."""
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        return {t for t in tokens if t not in self.STOPWORDS and len(t) > 2}

    def _get_text(self, art) -> str:
        """Extract text from article (dict or object)."""
        if isinstance(art, dict):
            return (art.get("title","") + " " + art.get("abstract",""))
        return (getattr(art,"title","") + " " + getattr(art,"abstract",""))

    def _get_meta(self, art) -> tuple:
        """Get (pmid, title, url) from article."""
        if isinstance(art, dict):
            return (str(art.get("pmid","")),
                    str(art.get("title","")),
                    str(art.get("url","")))
        return (str(getattr(art,"pmid","")),
                str(getattr(art,"title","")),
                str(getattr(art,"url","")))

    def _build_passages(self, articles: list) -> list:
        """Build list of passage dicts from articles."""
        passages = []
        for art in articles:
            text  = self._get_text(art)
            pmid, title, url = self._get_meta(art)
            # Split abstract into ~2-sentence passages
            sents = self._split_sentences(text)
            for i in range(0, len(sents), 2):
                chunk = " ".join(sents[i:i+2])
                passages.append({
                    "pmid":    pmid,
                    "title":   title,
                    "url":     url,
                    "passage": chunk,
                    "tokens":  self._clinical_tokens(chunk),
                })
        return passages

    def _jaccard(self, a: set, b: set) -> float:
        if not a or not b: return 0.0
        return len(a & b) / len(a | b)

    def _find_best_match(self, sentence: str, passages: list) -> GroundedSentence:
        """Find best matching passage for a sentence."""
        sent_tokens = self._clinical_tokens(sentence)
        if not sent_tokens:
            return GroundedSentence(text=sentence, is_grounded=False,
                                    overlap_score=0.0,
                                    source_passage="No tokens extracted")

        best_score   = 0.0
        best_passage = None

        for p in passages:
            score = self._jaccard(sent_tokens, p["tokens"])
            if score > best_score:
                best_score   = score
                best_passage = p

        is_grounded = best_score >= self.GROUNDED_THRESHOLD

        if best_passage:
            # Extract the most relevant snippet (≤120 chars)
            passage_text = best_passage["passage"]
            snippet = passage_text[:120] + ("..." if len(passage_text) > 120 else "")
            return GroundedSentence(
                text=sentence,
                pmid=best_passage["pmid"],
                source_title=best_passage["title"][:80],
                source_passage=snippet,
                overlap_score=round(best_score, 3),
                is_grounded=is_grounded,
                url=best_passage["url"],
            )
        return GroundedSentence(
            text=sentence, is_grounded=False,
            overlap_score=0.0, source_passage="No matching passage found"
        )