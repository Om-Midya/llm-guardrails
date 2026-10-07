import pytest

from guardrails.checks.pii import PIIInputCheck, PIIOutputCheck
from guardrails.core import Context

REDACT_CASES = [
    ("My PAN is ABCDE1234F, please update it.", "<IN_PAN>"),
    ("Aadhaar 2345 6789 0124 for KYC.", "<IN_AADHAAR>"),
    ("Transfer to IFSC HDFC0001234.", "<IN_IFSC>"),
    ("My account number 123456789012 shows a wrong debit.", "<IN_BANK_ACCOUNT>"),
    ("Call me on +91 98765 43210.", "<IN_PHONE>"),
    ("Card 4111 1111 1111 1111 got declined.", "<CREDIT_CARD>"),
    ("Send the statement to ravi.k@example.com.", "<EMAIL_ADDRESS>"),
]
ALLOW_CASES = [
    "What is the minimum balance for a savings account?",
    "I deposited 25000 rupees on 12 March 2026.",
    "Is the FD rate 7.1 percent for 18 months?",
    "My card ending 4321 was declined yesterday.",
    "Branch code is 4 digits, right?",
]


@pytest.mark.parametrize("text,tag", REDACT_CASES)
async def test_input_redacts(text, tag):
    v = await PIIInputCheck().check(text, Context())
    assert v.action == "redact" and tag in v.rewritten_text, v


@pytest.mark.parametrize("text", ALLOW_CASES)
async def test_input_allows_numbers_that_are_not_pii(text):
    v = await PIIInputCheck().check(text, Context())
    assert v.action == "allow", v


async def test_block_mode():
    v = await PIIInputCheck(action="block").check("PAN ABCDE1234F", Context())
    assert v.action == "block"


async def test_aadhaar_checksum_rejects_random_12_digits():
    v = await PIIInputCheck(entities=["IN_AADHAAR"]).check("number 1234 5678 9012", Context())
    assert v.action == "allow"


async def test_output_check_is_fail_closed_and_redacts():
    assert PIIOutputCheck.fail_closed is True
    v = await PIIOutputCheck().check("Customer email is a@b.co", Context())
    assert v.action == "redact" and "<EMAIL_ADDRESS>" in v.rewritten_text
