from abc import ABC, abstractmethod

from connector.market.market_manager import MarketManager, OrderBook, Trade, LastPrice, Candle
from connector.user.user_manager import UserManager
from connector.info.info_manager import InfoManager


class Strategy(ABC):
    """
    mm -- MarketManager (market data)
    um -- UserManager (placing orders)
    im -- InfoManager (instrument info)
    """

    def __init__(self):
        self.mm: MarketManager | None = None
        self.um: UserManager | None = None
        self.im: InfoManager | None = None

    def set_helpers(self, mm: MarketManager, um: UserManager, im: InfoManager):
        """
        Set MarketManager, UserManager and InfoManager
        This callback is invoked first
        """
        self.mm = mm
        self.um = um
        self.im = im

    @abstractmethod
    def subscribe(self):
        """
        This callback is invoked after "set_mm_um"
        1) Subscribe for market data
        2) Specify market data history to download
        """

    @abstractmethod
    def on_start(self):
        """
        1) All subscriptions are active
        2) All market data history is downloaded
        Then, this callback is invoked
        """

    @abstractmethod
    def on_order_book_update(self, order_book: OrderBook):
        """
        On market OrderBook Update
        """

    @abstractmethod
    def on_market_trade(self, trade: Trade):
        """
        On others Trade (very likely not our trade)
        """

    @abstractmethod
    def on_last_price(self, last_price: LastPrice):
        """
        On last price event (likely due to a trade)
        """

    @abstractmethod
    def on_candle(self, candle: Candle):
        """
        On new candle arrival (with certain interval)
        """

    @abstractmethod
    def on_order_event(self, order):
        """
        1) Error while placing order
        2) Market order is executed
        3) Limit order is placed
        4) Limit order is partially filled
        5) Limit order is filled
        """
