import pytest

from guardrails.checks.secret_leak import SecretLeakCheck, shannon_entropy
from guardrails.core import Context

POSITIVE = [
    "Use key AKIAIOSFODNN7EXAMPLE to access the bucket.",
    "Google key: AIzaSyD-9tSrke72PouQMnMX-a7eZSW0jkFMBWY",
    "token ghp_16C7e42F292c6912E7710c838347Ae178B4a",
    "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0."
    "dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U",
    "-----BEGIN RSA PRIVATE KEY-----\nMIIEow...",
    "api_key = 'sk_live_51H8f9K2eZvKYlo2'",
    "The internal password is: Tr0ub4dor&3xJ9!qLpW2zRvK",
]
NEGATIVE = [
    "Your reference number is NB2026031200451.",
    "Set a strong password with letters, numbers, and symbols.",
    "The IFSC code format is four letters, a zero, and six characters.",
    "Transaction ID TXN00012345 is pending.",
    "Visit https://www.novabank.example/cards for details.",
]


@pytest.mark.parametrize("text", POSITIVE)
async def test_blocks_secrets(text):
    v = await SecretLeakCheck().check(text, Context())
    assert v.action == "block", v.reason


@pytest.mark.parametrize("text", NEGATIVE)
async def test_allows_benign(text):
    v = await SecretLeakCheck().check(text, Context())
    assert v.action == "allow", v.reason


def test_entropy():
    assert shannon_entropy("aaaa") == 0
    assert shannon_entropy("Tr0ub4dor&3xJ9!qLpW2zRvK") > 4.0


async def test_zero_width_inside_key_is_still_caught():
    v = await SecretLeakCheck().check("key AKIA​IOSFODNN7EXAMPLE here", Context())
    assert v.action == "block", v.reason
