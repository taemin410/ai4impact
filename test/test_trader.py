import pytest
import os
from src.trade.trader import *

TEST_LEAD_TIME = 0
TRADER_SCENARIO1 = trader([10, 10, 10, 10, 10], [10, 10, 10, 10, 10], TEST_LEAD_TIME)
TRADER_SCENARIO2 = trader([10, 10, 10, 10, 10], [8, 8, 8, 8, 8], TEST_LEAD_TIME)
TRADER_SCENARIO3 = trader([10, 10, 10, 10, 10], [12, 12, 12, 12, 12], TEST_LEAD_TIME)
TRADER_SCENARIO4 = trader([10, 10], [210, 210], TEST_LEAD_TIME)

@pytest.mark.parametrize("trade_scenario,expected_cash_after_trade", [
    (TRADER_SCENARIO1, 1500),
    (TRADER_SCENARIO2, 1400),
    (TRADER_SCENARIO3, 1400),
    (TRADER_SCENARIO4, -30800)
])
def test_trade(trade_scenario, expected_cash_after_trade):
    assert trade_scenario.trade() == expected_cash_after_trade

