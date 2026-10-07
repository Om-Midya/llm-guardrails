from guardrails.checks.pii import build_analyzer
from guardrails.models import embedder, injection_classifier, toxicity_classifier

injection_classifier()
toxicity_classifier()
embedder()
build_analyzer()
print("models ready")
