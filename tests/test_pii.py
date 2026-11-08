from __future__ import annotations

import pytest

from memsvc import pii

KEY = b"k" * 32


@pytest.mark.parametrize(
    ("text", "kind"),
    [
        ("mail me at ana.silva+work@example.co.uk tomorrow", "EMAIL"),
        ("card 4111 1111 1111 1111 exp 12/29", "CARD"),
        ("my IBAN is GB82 WEST 1234 5698 7654 32", "IBAN"),
        ("ssn 123-45-6789", "SSN"),
        ("call +44 20 7946 0958 after 5", "PHONE"),
        ("call (415) 555-0132", "PHONE"),
        ("text me on +44 7700 900 123.", "PHONE"),  # at the end of a sentence
        ("server at 203.0.113.42 is down", "IP"),
        ("use sk-proj-abcdefghijklmnopqrstuvwx1234 for the API", "SECRET"),
        ("my password is hunter2!", "SECRET"),
        ("AKIAIOSFODNN7EXAMPLE leaked", "SECRET"),
    ],
)
def test_detects(text, kind):
    assert [f.kind for f in pii.find(text)] == [kind]


@pytest.mark.parametrize(
    "text",
    [
        "order 4111 1111 1111 1112 shipped",  # fails Luhn
        "version 10.2.3 released on 2026-03-04",
        "999.12.3.4 is not an address",
        "SSN-like 000-12-3456 is invalid",
        "the meeting is at 10:30 in room 4",
        "it costs $1,299.00 or 3 payments",
        "localhost is 127.0.0.1",
    ],
)
def test_ignores_lookalikes(text):
    assert pii.find(text) == []


def test_bad_iban_checksum_is_not_an_iban():
    assert "IBAN" not in {f.kind for f in pii.find("GB82 WEST 1234 5698 7654 33")}


def test_scrub_actions_and_stable_pseudonyms():
    s = pii.scrub("Email ana@example.com or ANA@example.com, phone +1 415 555 0132, from 203.0.113.9", KEY)
    a, b = s.text.split(" or ")[0].split()[-1], s.text.split(" or ")[1].split(",")[0]
    assert a == b and a.startswith("[EMAIL:")
    assert "[PHONE:" in s.text and "[IP]" in s.text and "ana@" not in s.text
    assert s.kinds == {"EMAIL": 2, "PHONE": 1, "IP": 1} and s.blocked == []
    other = pii.scrub("ana@example.com", b"x" * 32).text
    assert other != a  # keyed per tenant


def test_block_and_keep():
    s = pii.scrub("pay with 4111-1111-1111-1111", KEY)
    assert s.blocked == ["CARD"]
    s = pii.scrub("ana@example.com", KEY, {"EMAIL": "keep"})
    assert s.text == "ana@example.com"


def test_checksums():
    assert pii.luhn("4111111111111111") and not pii.luhn("4111111111111112")
    assert pii.iban_ok("DE89 3704 0044 0532 0130 00") and not pii.iban_ok("DE89 3704 0044 0532 0130 01")
    assert not pii.ssn_ok("666-12-3456") and pii.ssn_ok("123-45-6789")


def test_issuer_prefixes():
    assert pii.card_ok("378282246310005")  # Amex, 15 digits
    assert not pii.card_ok("1234567812345670")  # passes Luhn, no network issues 12...


def test_benchmark_scores():
    from memsvc.evaluation import pii_benchmark

    scores = pii_benchmark()
    assert all(v["recall"] == 1.0 for v in scores.values())
    assert min(v["precision"] for v in scores.values()) > 0.95
