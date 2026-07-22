"""The cost model is configuration, so it is worth a test.

Token *counts* come from the API and are exact; the money figure is only as
right as the rate table and the arithmetic over it. These tests pin the
arithmetic — in particular the two things people get wrong: cached prompt tokens
bill at the cheap cached rate, and reasoning ("thought") tokens bill at the
output rate even though they never appear in the answer.
"""

from src.llm import DEFAULT_RATE, RATES_USD_PER_MTOK, Usage


def _meta(prompt=0, out=0, thoughts=0, cached=0):
    return {
        "promptTokenCount": prompt,
        "candidatesTokenCount": out,
        "thoughtsTokenCount": thoughts,
        "cachedContentTokenCount": cached,
    }


def test_basic_cost_matches_the_rate_table():
    u = Usage()
    u.add("step", "gemini-3.5-flash", _meta(prompt=1000, out=200, thoughts=50, cached=400))
    r = RATES_USD_PER_MTOK["gemini-3.5-flash"]
    fresh_prompt = 1000 - 400
    expected = (fresh_prompt * r["input"] + 400 * r["cached"] + (200 + 50) * r["output"]) / 1e6
    assert u.cost_usd == expected
    assert u.cost_cents == expected * 100


def test_cached_tokens_are_cheaper_than_fresh_ones():
    # The whole reason caching is worth wiring up: the same prompt costs less
    # when it is served from cache.
    cached = Usage()
    cached.add("s", "gemini-3.5-flash", _meta(prompt=1000, out=100, cached=800))
    fresh = Usage()
    fresh.add("s", "gemini-3.5-flash", _meta(prompt=1000, out=100, cached=0))
    assert cached.cost_usd < fresh.cost_usd


def test_thought_tokens_bill_at_the_output_rate():
    r = RATES_USD_PER_MTOK["gemini-3.5-flash"]
    without = Usage()
    without.add("s", "gemini-3.5-flash", _meta(prompt=100, out=100, thoughts=0))
    with_ = Usage()
    with_.add("s", "gemini-3.5-flash", _meta(prompt=100, out=100, thoughts=40))
    # The delta is exactly 40 output-rate tokens — thoughts are not free.
    assert round(with_.cost_usd - without.cost_usd, 12) == round(40 * r["output"] / 1e6, 12)
    assert with_.thought_tokens == 40


def test_accumulates_across_calls():
    u = Usage()
    u.add("a", "gemini-3.5-flash", _meta(prompt=100, out=10))
    u.add("b", "gemini-3.5-flash", _meta(prompt=200, out=20))
    assert u.calls == 2
    assert u.prompt_tokens == 300
    assert u.output_tokens == 30
    assert len(u.per_step) == 2
    assert u.summary()["calls"] == 2


def test_unknown_model_falls_back_to_default_rate():
    u = Usage()
    u.add("s", "some-future-model", _meta(prompt=1000, out=100))
    expected = (1000 * DEFAULT_RATE["input"] + 100 * DEFAULT_RATE["output"]) / 1e6
    assert u.cost_usd == expected


def test_missing_fields_are_treated_as_zero_not_crash():
    u = Usage()
    u.add("s", "gemini-3.5-flash", {})  # nothing reported
    assert u.cost_usd == 0.0
    assert u.calls == 1
