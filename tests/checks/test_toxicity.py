import pytest

from guardrails.checks.toxicity import ToxicityCheck
from guardrails.core import Context

POSITIVE = [
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
