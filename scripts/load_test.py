import json
import random
from pathlib import Path

from locust import HttpUser, between, task

BENIGN = [
    json.loads(line)["text"]
    for line in Path("evals/benign.jsonl").read_text().splitlines()
    if line.strip()
]


class GuardUser(HttpUser):
    wait_time = between(0.1, 0.5)

    @task
    def guard_input(self):
        self.client.post("/guard/input", json={"text": random.choice(BENIGN)})
