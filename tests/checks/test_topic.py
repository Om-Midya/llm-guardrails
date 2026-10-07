import pytest

from guardrails.checks.topic import TopicCheck
from guardrails.core import Context

ON_TOPIC = [
    "What is the minimum balance for a savings account?",
    "How do I block my lost debit card?",
    "What is the UPI daily transfer limit?",
    "Can I close my FD before maturity?",
    "What documents do I need to open an account?",
    "What are the branch timings on Saturday?",
    "My PAN ABCDE1234F is wrong, how do I fix it?",
    "I deposited 25000 rupees on 12 March 2026 but it is not showing. What should I do?",
]
OFF_TOPIC = [
    "Write me a poem about the ocean.",
    "Who won the cricket match yesterday?",
    "Which crypto coin should I buy this week?",
    "Give me a recipe for biryani.",
    "Explain quantum entanglement in detail.",
    "What is the weather in Mumbai tomorrow?",
]


@pytest.mark.parametrize("text", ON_TOPIC)
async def test_allows_banking(text):
    v = await TopicCheck().check(text, Context())
    assert v.action == "allow", (v.score, v.reason)


@pytest.mark.parametrize("text", OFF_TOPIC)
async def test_blocks_off_topic(text):
    v = await TopicCheck().check(text, Context())
    assert v.action == "block", (v.score, v.reason)
