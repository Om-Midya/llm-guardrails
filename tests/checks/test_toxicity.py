import pytest

from guardrails.checks.toxicity import ToxicityCheck
from guardrails.core import Context

POSITIVE = [
    '{"answer": "Only a complete moron would miss an EMI three times. Pay up or get lost.", '
    '"sources": ["loan-emi-missed.md"], "confidence": 0.6}',
    "You are a worthless idiot and your complaint is garbage.",
    "Shut up, stupid customer, nobody cares about your money.",
    "I hope your account gets hacked, you moron.",
]
NEGATIVE = [
    "Your card will be blocked after three wrong PIN attempts.",
    "I am sorry for the trouble. The dispute will be resolved in 7 working days.",
    "Fraudulent transactions are reversed after investigation.",
    "This fee is non-refundable as per the schedule of charges.",
]


@pytest.mark.parametrize("text", POSITIVE)
async def test_blocks_toxic(text):
    v = await ToxicityCheck().check(text, Context())
    assert v.action == "block", (v.score, v.reason)


@pytest.mark.parametrize("text", NEGATIVE)
async def test_allows_firm_but_polite(text):
    v = await ToxicityCheck().check(text, Context())
    assert v.action == "allow", (v.score, v.reason)


async def test_long_output_truncated():
    v = await ToxicityCheck().check("Thank you for banking with us. " * 1500, Context())
    assert v.action == "allow"


async def test_toxic_tail_after_long_padding_is_caught(monkeypatch):
    from guardrails.checks import toxicity as tx

    seen = []

    def fake(text):
        seen.append(len(text))
        return ("toxic", 0.95) if "moron" in text else ("toxic", 0.01)

    monkeypatch.setattr(tx.ToxicityCheck, "_score", staticmethod(fake))
    text = "Thank you for banking with us. " * 150 + "You absolute moron."
    v = await ToxicityCheck().check(text, Context())
    assert v.action == "block" and len(seen) >= 2 and max(seen) <= 1500
