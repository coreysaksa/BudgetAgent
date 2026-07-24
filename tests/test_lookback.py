from budget_agent.lookback import (
    DEFAULT_LOOKBACK_DAYS,
    MAX_LOOKBACK_DAYS,
    parse_lookback_days,
)


def test_defaults_to_30_when_no_window_mentioned():
    assert parse_lookback_days("how am I doing this month?") == DEFAULT_LOOKBACK_DAYS
    assert parse_lookback_days("") == DEFAULT_LOOKBACK_DAYS
    assert parse_lookback_days("where can I save money?") == DEFAULT_LOOKBACK_DAYS


def test_explicit_day_windows():
    assert parse_lookback_days("looking back 60 days") == 60
    assert parse_lookback_days("show me the last 90 days of spending") == 90
    assert parse_lookback_days("over the past 45 days") == 45
    assert parse_lookback_days("give me a 14-day summary") == 14


def test_weeks_months_years():
    assert parse_lookback_days("past 2 weeks") == 14
    assert parse_lookback_days("looking back 6 months") == 180
    assert parse_lookback_days("over the last 3 months") == 90
    assert parse_lookback_days("look back a year") == 365
    assert parse_lookback_days("spending over the last year") == 365


def test_spelled_out_numbers():
    assert parse_lookback_days("past six months") == 180
    assert parse_lookback_days("the last three weeks") == 21
    assert parse_lookback_days("look back two years") == 730


def test_bare_keyword_unit_defaults_to_one():
    assert parse_lookback_days("how did I do last month?") == 30
    assert parse_lookback_days("summarize the past week") == 7
    assert parse_lookback_days("what about last year?") == 365


def test_clamped_to_max():
    assert parse_lookback_days("look back 10 years") == MAX_LOOKBACK_DAYS
    assert parse_lookback_days("past 5000 days") == MAX_LOOKBACK_DAYS


def test_does_not_false_positive_on_casual_day_mentions():
    # No lookback keyword and no explicit quantity+unit -> keep the default.
    assert parse_lookback_days("have a good day!") == DEFAULT_LOOKBACK_DAYS
    assert parse_lookback_days("is my monthly budget ok?") == DEFAULT_LOOKBACK_DAYS
    assert parse_lookback_days("what's my yearly income?") == DEFAULT_LOOKBACK_DAYS


def test_first_window_wins_when_multiple():
    # Explicit quantity+unit is the strong signal and is taken first.
    assert parse_lookback_days("compare the last 60 days to prior month") == 60
