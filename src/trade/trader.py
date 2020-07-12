
IINITIAL_CASH_OFFERINGS = 1000
SELL_PRICE = 10
BUY_PRICE = 20
FINE = 100

class trader():
    def __init__(self, real_data, forecast_data, lead_time=18):
        assert len(real_data) == len(forecast_data)

        self.lead_time = lead_time
        self._cash_at_hand = IINITIAL_CASH_OFFERINGS 
        self._real_data = real_data 
        self._forecast_data = forecast_data
    
    def trade(self) -> float:
        for i, (real_val, forecast_val) in enumerate(zip(self._real_data, self._forecast_data)):
            if i < self.lead_time:  # warm-up
                continue
            
            if forecast_val < real_val: # excess
                self._manage_excess()
            elif forecast_val > real_val:   # shortfall
                self._manage_shortfall(real_val, forecast_val)          

            self._sell(forecast_val)

        return self._cash_at_hand

    def pay_back(self) -> float:
        self._cash_at_hand -= IINITIAL_CASH_OFFERINGS

    def _sell(self, forecast_val) -> None:
        earnings = forecast_val * SELL_PRICE
        self._cash_at_hand += earnings

    def _manage_excess(self) -> None:
        pass

    def _manage_shortfall(self, real_val, forecast_val) -> None:
        shortfall_energy = forecast_val - real_val
        shortfall_money = shortfall_energy * BUY_PRICE
        loss = 0

        if shortfall_money > self._cash_at_hand:    # charge fine

            if self._cash_at_hand <= 0: # no money at hand to pay for the shortfall
                loss += shortfall_energy * FINE

            else:   # some of the shortfall payable by the cash at hand 
                non_payable_shortfall = shortfall_money - self._cash_at_hand
                payable_shortfall = shortfall_money - non_payable_shortfall
                loss += (non_payable_shortfall / BUY_PRICE * FINE) + payable_shortfall
                
        else:   # no fine, but pay for the shortfall
            loss += shortfall_money

        self._cash_at_hand -= loss

        

