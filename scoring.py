"""
Refine sentences with spaCy and score tone with FinBERT.

spaCy re-segments the raw commentary (better sentence boundaries than
regex, plus a cheap "is this in management voice?" filter). FinBERT
(ProsusAI/finbert, pre-trained on the Financial PhraseBank) then scores
each sentence; net_tone = positive - negative.

    python scoring.py                    # score all unscored rows in DB
    python scoring.py --ticker AAPL
    python scoring.py --batch-size 64    # bump on GPU
"""

from __future__ import annotations

import argparse
import functools
import os
from typing import Optional

import psycopg2.extras
from dotenv import load_dotenv

import db

load_dotenv()

FINBERT_MODEL = os.getenv("FINBERT_MODEL", "ProsusAI/finbert")
SPACY_MODEL = os.getenv("SPACY_MODEL", "en_core_web_sm")
DEVICE = int(os.getenv("DEVICE", "-1"))   # -1 = CPU, 0 = first GPU
# batch sizes above 32 can OOM on CPU with finbert — 16 is a safe default
DEFAULT_BATCH_SIZE = 16


@functools.lru_cache(maxsize=1)
def _get_spacy():
    import spacy  # type: ignore
    print(f"[spacy] Loading model: {SPACY_MODEL} …")
    try:
        return spacy.load(SPACY_MODEL)
    except OSError:
        print(f"[spacy] Model not found. Run: python -m spacy download {SPACY_MODEL}")
        raise


@functools.lru_cache(maxsize=1)
def _get_finbert():
    from transformers import pipeline  # type: ignore
    print(f"[finbert] Loading model: {FINBERT_MODEL} …")
    return pipeline(
        "text-classification",
        model=FINBERT_MODEL,
        tokenizer=FINBERT_MODEL,
        top_k=None,           # return all three label scores
        device=DEVICE,
        max_length=512,
        truncation=True,
    )


def refine_sentences_spacy(raw_sentences: list[str]) -> list[str]:
    """
    Re-segment raw sentences through spaCy for accurate boundaries
    and filter to high-signal management commentary sentences.
    """
    try:
        nlp = _get_spacy()
    except OSError:
        print(f"[spacy] Model '{SPACY_MODEL}' not downloaded — returning raw sentences unchanged")
        return raw_sentences

    refined: list[str] = []
    for text in raw_sentences:
        doc = nlp(text[:1000])   # cap for performance
        for sent in doc.sents:
            s = sent.text.strip()
            if _is_management_voice(sent) and len(s) > 20:
                refined.append(s)
    return refined


def _is_management_voice(span) -> bool:
    """Heuristic: keep sentences that look like management commentary.

    Forward-looking language and financial keywords are the signal.
    False positive rate is acceptable — FinBERT scoring handles noise downstream.
    """
    import spacy  # type: ignore

    text_lower = span.text.lower()
    financial_keywords = {
        "revenue", "growth", "earnings", "margin", "guidance", "outlook",
        "quarter", "fiscal", "income", "profit", "loss", "cash flow",
        "we expect", "we believe", "we continue", "we remain",
    }
    return any(kw in text_lower for kw in financial_keywords)


def score_sentences(sentences: list[str]) -> list[dict]:
    """Score sentences with FinBERT. Returns dicts with positive/negative/neutral scores.

    Note: FinBERT returns probabilities summing to ~1.0 per sentence. net_tone
    downstream is computed as positive_score - negative_score so it's in [-1, 1].
    """
    if not sentences:
        return []
    pipe = _get_finbert()
    results = pipe(sentences)
    scored = []
    for r in results:
        scores = {item["label"].lower(): item["score"] for item in r}
        scored.append({
            "positive_score": scores.get("positive", 0.0),
            "negative_score": scores.get("negative", 0.0),
            "neutral_score":  scores.get("neutral",  0.0),
        })
    return scored


def _get_unscored_commentary(ticker: Optional[str], batch_size: int) -> list[dict]:
    """Fetch commentary rows that haven't been scored yet."""
    where = "AND c.ticker = %(ticker)s" if ticker else ""
    sql = f"""
    SELECT c.id, c.ticker, c.sentence_text, f.filed_at
    FROM commentary c
    JOIN filings f ON f.id = c.filing_id
    WHERE NOT EXISTS (
        SELECT 1 FROM tone_scores ts WHERE ts.commentary_id = c.id
    )
    {where}
    ORDER BY f.filed_at DESC
    LIMIT %(limit)s
    """
    with db.get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, {"ticker": ticker, "limit": batch_size})
            return [dict(r) for r in cur.fetchall()]


def run_scoring(ticker: Optional[str] = None, batch_size: int = 64):
    """Score all unscored commentary for a ticker (or all tickers)."""
    total_scored = 0

    while True:
        batch = _get_unscored_commentary(ticker, batch_size)
        if not batch:
            print(f"[scoring] No unscored sentences remaining. Total scored: {total_scored}")
            break

        sentences = [row["sentence_text"] for row in batch]
        # Optionally refine via spaCy
        try:
            refined = refine_sentences_spacy(sentences)
        except Exception as e:
            print(f"[spacy] Refinement failed ({e}), using raw sentences")
            refined = sentences

        # Score with FinBERT
        scores = score_sentences(refined or sentences)

        # Build DB records
        tone_records = []
        for row, score in zip(batch, scores):
            tone_records.append({
                "commentary_id": row["id"],
                "ticker": row["ticker"],
                "filed_at": row["filed_at"],
                **score,
            })

        db.insert_tone_scores(tone_records)
        total_scored += len(tone_records)
        print(f"[scoring] Scored {len(tone_records)} sentences (total: {total_scored})")

        if len(batch) < batch_size:
            break

    return total_scored


def main():
    parser = argparse.ArgumentParser(description="Score management commentary with FinBERT")
    parser.add_argument("--ticker", default=None, help="Limit to a single ticker")
    parser.add_argument("--batch-size", type=int, default=64, help="Sentences per batch")
    args = parser.parse_args()
    run_scoring(ticker=args.ticker, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
