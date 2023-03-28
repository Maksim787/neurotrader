import datetime

from connector.market.market_manager import Trade, OrderBook, LastPrice, Candle
from connector.user.user_manager import Orders, Positions, NewOrder, ModifyOrder, UserOrder, OrderType, OrderDirection, CancelOrder
from connector.info.info_manager import InstrumentInfo
from connector.strategy import Strategy
from connector.common.log import Logging


class MyStrategy(Strategy):
    QUANTITY = 1

    def __init__(self, ticker: str) -> None:
        super().__init__()
        self.logger = Logging.get_logger('MyStrategy')
        self.ticker = ticker
        self.instrument: InstrumentInfo | None = None
        self.orders: Orders | None = None
        self.positions: Positions | None = None
        self.bid_order: UserOrder | None = None
        self.ask_order: UserOrder | None = None

    def subscribe(self) -> None:
        self.instrument = self.im.get_instrument_by_ticker(self.ticker)
        self.mm.subscribe_order_book([self.instrument], depth=1)
        self.rn.subscribe_interval(datetime.timedelta(seconds=5))

    def on_start(self) -> None:
        self.logger.info('OnStart is called')
        self.positions = self.um.get_positions()
        self.orders = self.um.get_orders()

    def on_order_book_update(self, order_book: OrderBook) -> None:
        """
        Place quotes on best bid/ask
        """
        self.logger.info(f'OnOrderBookUpdate is called with {order_book}')
        best_bid = order_book.bids[0]
        best_ask = order_book.asks[0]
        mid_price = order_book.mid_price
        self.logger.info(f'Best bid/ask: {best_bid = }; {best_ask = }; {mid_price = }')
        for order, direction in zip([self.bid_order, self.ask_order], [OrderDirection.BUY, OrderDirection.SELL]):
            if len(self.orders.pending_orders) > 0:
                continue
            if order is not None:
                self._modify_existing_order(order_book, order)
            else:
                self._place_new_order(order_book, direction)

    def _place_new_order(self, order_book: OrderBook, direction: OrderDirection):
        # Check blocked and available money
        if self.positions.money < order_book.mid_price and direction == OrderDirection.BUY:
            return
        if self.positions.securities[self.instrument.figi].balance < 1 and direction == OrderDirection.SELL:
            return

        price = order_book.best_bid_price if direction == OrderDirection.BUY else order_book.best_ask_price
        qty = self.QUANTITY
        self.logger.info(f'Place order: [{direction = }; {price = }; {qty = }]')
        self.logger.info(f'Positions: {self.positions.description()}')
        self.logger.info(f'Orders: {self.orders.description()}')
        self.um.new_order(
            NewOrder(
                instrument=self.instrument,
                order_type=OrderType.LIMIT,
                direction=direction,
                quantity=qty,
                price=price
            )
        )

    def _modify_existing_order(self, order_book: OrderBook, user_order: UserOrder):
        direction = user_order.direction
        price = order_book.best_bid_price if direction == OrderDirection.BUY else order_book.best_ask_price
        if user_order.price == price:
            return
        qty = self.QUANTITY
        self.logger.info(f'Cancel order order and wait to place new one: [{direction = }; {price = }; {qty = }]')
        self.logger.info(f'Positions: {self.positions.description()}')
        self.logger.info(f'Orders: {self.orders.description()}')
        self.um.cancel_order(
            CancelOrder(
                order_id=user_order.order_id
            )
        )
        # self.um.modify_order(
        #     ModifyOrder(
        #         order_id=user_order.order_id,
        #         new_quantity=qty,
        #         new_price=price
        #     )
        # )

    def on_order_event(self, order: UserOrder) -> None:
        """
        Update internal state
        """
        order_field = 'bid_order' if order.direction == OrderDirection.BUY else 'ask_order'
        if order.status.is_closed():
            self.logger.info(f'{order_field} is closed')
            setattr(self, order_field, None)
        else:
            self.logger.info(f'{order_field} is open')
            setattr(self, order_field, order)
        self.logger.info(f'OnOrderEvent: {order}')
        self.logger.info(f'Positions: {self.positions.description()}')
        self.logger.info(f'Orders: {self.orders.description()}')

    def on_market_trade(self, trade: Trade) -> None:
        """
        Is not used
        """
        self.logger.info(f'OnTradeUpdate is called with {trade}')

    def on_last_price(self, last_price: LastPrice) -> None:
        """
        Is not used
        """
        self.logger.info(f'OnLastPriceUpdate is called with {last_price}')

    def on_candle(self, candle: Candle) -> None:
        """
        Is not used
        """
        self.logger.info(f'OnCandleUpdate is called with {candle}')

    def on_interval(self, interval: datetime.timedelta) -> None:
        """
        Is not used
        """
        self.logger.info(f'OnInterval: {interval.total_seconds()}')
