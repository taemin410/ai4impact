from datetime import datetime
from src.utils.logger import logger

EXCESS_TRADE = "excess"
SHORTFALL_TRADE = "shortfall"
EXACT_TRADE = "exact"

IINITIAL_CASH_OFFERINGS = 1000
SELL_PRICE = 10
BUY_PRICE = 20
FINE = 100


class trader:
    def __init__(self, real_data, forecast_data, lead_time=18):
        assert len(real_data) == len(forecast_data)

        self.lead_time = lead_time
        self._cash_at_hand = IINITIAL_CASH_OFFERINGS
        self._real_data = real_data
        self._forecast_data = forecast_data

    def trade(self) -> float:
        """
            Deals with different trade scenario based on forecast and real value.
        """
        logger_ = logger(datetime.now())

        for i, (real_val, predict_val) in enumerate(
            zip(self._real_data, self._forecast_data)
        ):
            if i < self.lead_time:  # warm-up
                continue

            trade_type, diff, lost_val = self._initialize_trade()

            if predict_val < real_val:  # excess
                trade_type = EXCESS_TRADE
                diff, lost_val = self._manage_excess(real_val, predict_val)

            elif predict_val > real_val:  # shortfall
                trade_type = SHORTFALL_TRADE
                diff, lost_val = self._manage_shortfall(real_val, predict_val)

            self._sell(predict_val)

            logger_.append_trade_data(
                i, trade_type, real_val, predict_val, diff, lost_val, self._cash_at_hand
            )

        logger_.log_trade_history()
        return self._cash_at_hand

    def pay_back(self) -> float:
        """
            Pay back the initial money offered.
        """
        self._cash_at_hand -= IINITIAL_CASH_OFFERINGS
        return self._cash_at_hand

    def _initialize_trade(self) -> tuple:
        return EXACT_TRADE, 0, 0

    def _sell(self, predict_val) -> None:
        earnings = predict_val * SELL_PRICE
        self._cash_at_hand += earnings

    def _manage_excess(self, real_val, predict_val) -> tuple:
        excess_energy = real_val - predict_val
        loss = excess_energy * 10
        return excess_energy, loss

    def _manage_shortfall(self, real_val, predict_val) -> tuple:
        shortfall_energy = predict_val - real_val
        shortfall_money = shortfall_energy * BUY_PRICE
        cost = 0

        if shortfall_money > self._cash_at_hand:  # charge fine

            if self._cash_at_hand <= 0:  # no money at hand to pay for the shortfall
                cost += shortfall_energy * FINE

            else:  # some of the shortfall payable by the cash at hand
                non_payable_shortfall = shortfall_money - self._cash_at_hand
                payable_shortfall = shortfall_money - non_payable_shortfall
                cost += (non_payable_shortfall / BUY_PRICE * FINE) + payable_shortfall

        else:  # no fine, but pay for the shortfall
            cost += shortfall_money

        self._cash_at_hand -= cost
        return shortfall_energy, cost
