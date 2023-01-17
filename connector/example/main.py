import asyncio
import yaml
import time

from connector.market.market_manager import Trade, OrderBook, LastPrice, Candle, CandleInterval
from connector.strategy import Strategy
from connector.runner import Runner
from connector.info.info_manager import InstrumentInfo
from connector.common.log import Logging


class MyStrategy(Strategy):
    def __init__(self, ticker: str):
        super().__init__()
        self._logger = Logging.get_logger('MyStrategy')
        self.ticker = ticker
        self.instrument: InstrumentInfo | None = None

    def subscribe(self):
        self.instrument = self.im.get_instrument_by_ticker(self.ticker)
        # self.mm.subscribe_order_book([self.instrument], depth=1)
        self.mm.subscribe_trades([self.instrument])
        # self.mm.subscribe_last_prices([self.instrument])
        self.mm.subscribe_candles([self.instrument], interval=CandleInterval.MIN_1)

    def on_start(self):
        self._logger.info('OnStart is called')

    def on_order_book_update(self, order_book: OrderBook):
        self._logger.info(f'OnOrderBookUpdate is called with {order_book}')
        time.sleep(1e-3)

    def on_market_trade(self, trade: Trade):
        self._logger.info(f'OnTradeUpdate is called with {trade}')
        time.sleep(1e-3)

    def on_last_price(self, last_price: LastPrice):
        self._logger.info(f'OnLastPriceUpdate is called with {last_price}')
        time.sleep(1e-3)

    def on_candle(self, candle: Candle):
        self._logger.info(f'OnCandleUpdate is called with {candle}')
        time.sleep(1e-3)

    def on_order_event(self, order):
        pass


async def main():
    logger.info('Create Runner and Strategy')
    # Read token
    with open('connector/private/keys.yaml') as f:
        private_keys = yaml.safe_load(f)
        token = private_keys['token']
    # Create strategy
    strategy = MyStrategy('TMOS')
    logger.info('Run Strategy')
    async with Runner(token) as runner:
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
