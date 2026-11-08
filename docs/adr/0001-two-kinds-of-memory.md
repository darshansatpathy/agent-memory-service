# 1. Two kinds of memory: episodes that fade, facts that change

Status: accepted

## Context

What an agent should remember about a person comes in two shapes. Episodes ("we planned a trip to Peru in May") are many, become less relevant with time, and should eventually go. Facts ("lives in Porto") are few, stay relevant, and change: when they change, the old value is wrong, not merely older.

Most memory stores treat everything as an append-only list of text chunks ranked by similarity. They return the old city as readily as the new one.

## Decision

- **Episodic** memories carry the time of the event, a default TTL of 90 days, and a recency weight at recall (half-life 30 days, never below 60% of the score).
- **Semantic** memories have no default TTL. A semantic memory may name a `subject` (`user.city`); writing a new value for the subject supersedes the previous one in the same transaction. Superseded memories are kept as history, excluded from recall by default, and returned with half weight when history is asked for.
- Semantic memories without a subject that are near-duplicates of an existing one (cosine ≥ 0.92) merge into it rather than accumulate.
- Exact repeats of any memory update `seen_count` and `last_seen_at` instead of adding a row.

## Consequences

Callers decide what a subject is. The service does not extract subjects from text; a wrong subject key supersedes the wrong fact, so keys belong in the agent's code, not in a model's output.
