import asyncio
import datetime

import yaml
import time

from connector.market.market_manager import Trade, OrderBook, LastPrice, Candle
from connector.user.user_manager import Orders, Positions, NewOrder, UserOrder, OrderType, OrderDirection
from connector.info.info_manager import InstrumentInfo
from connector.runner import Runner
from connector.strategy import Strategy
from connector.common.log import Logging


class MyStrategy(Strategy):
    def __init__(self, ticker: str) -> None:
        super().__init__()
        self.logger = Logging.get_logger('MyStrategy')
        self.ticker = ticker
        self.instrument: InstrumentInfo | None = None
        self.orders: Orders | None = None
        self.positions: Positions | None = None

    def subscribe(self) -> None:
        self.instrument = self.im.get_instrument_by_ticker(self.ticker)
        # self.mm.subscribe_order_book([self.instrument], depth=1)
        # self.mm.subscribe_trades([self.instrument])
        # self.mm.subscribe_last_prices([self.instrument])
        # self.mm.subscribe_candles([self.instrument], interval=CandleInterval.MIN_1)
        self.rn.subscribe_interval(datetime.timedelta(seconds=5))

    def on_start(self) -> None:
        self.logger.info('OnStart is called')
        self.um.new_order(
            NewOrder(
                instrument=self.instrument,
                order_type=OrderType.LIMIT,
                direction=OrderDirection.BUY,
                quantity=1,
                price=4.0
            )
        )

    def on_order_book_update(self, order_book: OrderBook) -> None:
        self.logger.info(f'OnOrderBookUpdate is called with {order_book}')
        time.sleep(1e-3)

    def on_market_trade(self, trade: Trade) -> None:
        self.logger.info(f'OnTradeUpdate is called with {trade}')
        time.sleep(1e-3)

    def on_last_price(self, last_price: LastPrice) -> None:
        self.logger.info(f'OnLastPriceUpdate is called with {last_price}')
        time.sleep(1e-3)

    def on_candle(self, candle: Candle) -> None:
        self.logger.info(f'OnCandleUpdate is called with {candle}')
        time.sleep(1e-3)

    def on_order_event(self, order: UserOrder) -> None:
        self.logger.info(f'OnOrderEvent: {order}')
        self.logger.info(f'Positions: {self.positions.description()}')
        self.logger.info(f'Orders: {self.orders.description()}')

    def on_interval(self, interval: datetime.timedelta) -> None:
        self.logger.info(f'OnInterval: {interval.total_seconds()}')


async def main() -> None:
    logger.info('Create Runner and Strategy')
    # Read token
    with open('connector/private/keys.yaml') as f:
        private_keys = yaml.safe_load(f)
        token = private_keys['token']
        account_id = private_keys['account_id']
        if account_id is not None:
            account_id = str(account_id)
    # Create strategy
    strategy = MyStrategy('TMOS')
    logger.info('Run Strategy')
    async with Runner(token, account_id) as runner:
        await runner.run(strategy)  # main loop


if __name__ == '__main__':
    logger = Logging.get_logger('main')
    try:
        asyncio.run(main())
    except KeyboardInterrupt as ex:
        logger.info('KeyboardInterrupt in main.py')
        logger.exception(ex)
        logger.info('Normal exit via KeyboardInterrupt')
    except Exception as ex:
        logger.error('Exception in main.py')
        logger.exception(ex)
        logger.error('Error exit')
    logger.info('Final Exit')
