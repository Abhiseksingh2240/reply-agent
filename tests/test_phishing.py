from app.scoring import score_phishing_text


def test_phishing_urgency_and_lookalike():
    text = "URGENT verify now http://paypa1-secure.net/verify"
    score, flags = score_phishing_text(text)
    assert score > 0.2
    assert any("urgency" in f for f in flags)
