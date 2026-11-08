# 2. Personal data is scrubbed before it is embedded or stored

Status: accepted

## Context

Agents hear personal data constantly. Whatever reaches a memory store reaches its backups, its embeddings and every future prompt it is recalled into. Embeddings of raw text can leak the text.

## Decision

- Detection runs before embedding, on the write path and on recall queries.
- Detectors are patterns with validators: Luhn and issuer prefixes for cards, mod-97 for IBANs, area rules for SSNs, address ranges for IPs, digit counts for phones, and known prefixes for API keys and tokens. A lookalike (an order number, a version string) is not a match.
- Actions per kind: **block** secrets, card numbers, IBANs and SSNs (the write fails with 422 and the kinds found); **pseudonymise** emails and phone numbers into stable keyed tokens (`[EMAIL:3f9a1c2e]`, HMAC with a per-tenant key), so the agent can still tell two mentions apart or together; **redact** IP addresses.
- What was found is stored as kinds and counts. Values never reach the database, logs or metrics.

## Consequences

Rule-based detection misses personal data with no pattern (a home address in prose, a name). The policy is a dictionary per deployment, and a model-based detector can be added behind the same interface.
