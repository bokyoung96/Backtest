from abc import ABC, abstractmethod


# NOTE: Investing & rebalancing scheme should all be in methodology.
# NOTE: start_date and end_date will be determined in methodology.
# NOTE: Methodologies have mkt, start_date, end_date as common.
class Methodology(ABC):
    def __init__(self,
                 mkt: str,
                 start_date: str,
                 end_date: str,
                 **kwargs):
        self.mkt = mkt
        self.start_date = start_date
        self.end_date = end_date

        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    @abstractmethod
    def weights(self):
        # NOTE: weights should have 1, -1, or NaN. (Not zero!)
        pass
