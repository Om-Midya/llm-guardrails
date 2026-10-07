from bankassist.retriever import Retriever


def test_search_returns_relevant_chunk():
    r = Retriever()
    hits = r.search("What is the daily UPI limit?", k=3)
    assert len(hits) == 3
    assert "upi.md" in [h.source for h in hits[:2]], [h.source for h in hits]
    assert hits[0].score >= hits[1].score >= hits[2].score


def test_sources_are_file_names():
    hits = Retriever().search("block lost debit card")
    assert all(h.source.endswith(".md") for h in hits)
    assert "card-lost-stolen.md" in [h.source for h in hits]
