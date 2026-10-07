from guardrails.models import MAX_WINDOWS, OVERLAP, WINDOW, windows


def test_short_text_is_one_window():
    assert windows("hello") == ["hello"]


def test_windows_overlap_and_fit_model():
    text = "a" * (WINDOW * 3)
    ws = windows(text)
    assert all(len(w) <= WINDOW for w in ws)
    assert ws[1][:OVERLAP] == ws[0][-OVERLAP:]
    assert "".join(w[OVERLAP:] if i else w for i, w in enumerate(ws)) == text


def test_window_count_capped():
    assert len(windows("x" * 1_000_000)) == MAX_WINDOWS
