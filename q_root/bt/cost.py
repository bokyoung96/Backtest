class TransactionCost:

    def __init__(self, buy_commission, sell_commission, slippage, sell_tax, cash_rate):
        self._buy_commission = buy_commission
        self._sell_commission = sell_commission
        self._slippage = slippage
        self._sell_tax = sell_tax
        self._cash_rate = cash_rate

    @property
    def buy_commission(self):
        return self._buy_commission

    @property
    def sell_commission(self):
        return self._sell_commission

    @property
    def slippage(self):
        return self._slippage

    @property
    def sell_tax(self):
        return self._sell_tax

    @property
    def cash_rate(self):
        return self._cash_rate

    def __str__(self):
        return (f"TransactionCost(buy_commission={self.buy_commission}, "
                f"sell_commission={self.sell_commission}, slippage={self.slippage}, "
                f"sell_tax={self.sell_tax}, cash_rate={self.cash_rate})")


class KoreaTransactionCost(TransactionCost):
    def __init__(self):
        super().__init__(buy_commission=0.0002, sell_commission=0.0002,
                         slippage=0.0005, sell_tax=0.0015, cash_rate=0.0000)


class NoTransactionCost(TransactionCost):
    def __init__(self):
        super().__init__(buy_commission=0.0000, sell_commission=0.0000,
                         slippage=0.0000, sell_tax=0.0000, cash_rate=0.0000)
