# 4. Recall: vector search by default, hybrid on request

Status: accepted

## Context

Vector search finds paraphrases ("which city is she based in" → "we relocated to Porto"). Full-text search finds rare exact words (a product name, a pet's name) that small embedding models blur. Neither is best everywhere.

## Decision

- Each recall runs an HNSW cosine search and a full-text search (an OR of the question's words) over the caller's namespace, 50 candidates each.
- The two are fused by relative score: cosine similarity and text rank are each scaled to 0..1 across the candidates and mixed 0.8 / 0.2. Importance (±10%) and, for episodes, recency (down to −15% with a 30-day half-life) then adjust the score. Each hit reports its vector rank, text rank and similarity, so a caller can see why it was returned.
- We tried reciprocal rank fusion first. Its scores for neighbouring ranks differ by under 2%, so any recency or importance weight overrode relevance entirely: in the benchmark no episode ever reached the top five. Relative score fusion keeps the gaps between candidates, so the weights only settle close calls.
- HNSW iterative scans (`hnsw.iterative_scan = relaxed_order`) keep the namespace filter from leaving the vector search with too few rows; the outer query re-sorts by distance.
- `mode` selects vector, text or hybrid per request; the benchmark reports all three. **Vector is the default** because it scored best overall (hit@1 0.969 against 0.955 for hybrid, `docs/evaluation.md`); hybrid ranks episodes slightly better and is there for traffic full of exact tokens that an embedder blurs.

## Consequences

Two searches per recall. At 100,000 memories the benchmark measures what that costs (`docs/evaluation.md`).
