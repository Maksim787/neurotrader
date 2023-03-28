import datetime
import typing as tp
import logging
from dataclasses import dataclass
from enum import Enum

import tinkoff.invest as inv
from tinkoff.invest.market_data_stream.async_market_data_stream_manager import AsyncMarketDataStreamManager
from tinkoff.invest.async_services import AsyncServices
from connector.info.info_manager import InfoManager, InstrumentInfo
from connector.common.log import Logging
from connector.common.time import utc_to_local, utc_now
from connector.common.quotation import quotation_to_float

if tp.TYPE_CHECKING:
    from connector.strategy import Strategy
    from connector.runner import Runner


####################################################################################################
# OrderBook
####################################################################################################

@dataclass
class Order:
    price: float  # price per instrument (not per lot)
    quantity: float  # number of lots


@dataclass
class OrderBook:
    instrument_info: InstrumentInfo
    depth: int
    bids: list[Order]
    asks: list[Order]
    time: datetime.datetime  # local time
    limit_up: float  # per instrument (not per lot)
    limit_down: float  # per instrument (not per lot)

    @property
    def mid_price(self) -> float:
        return (self.bids[0].price + self.asks[0].price) / 2

    @property
    def best_bid_price(self) -> float:
        return self.bids[0].price

    @property
    def best_ask_price(self) -> float:
        return self.asks[0].price


####################################################################################################
# Trade
####################################################################################################

class TradeDirection(Enum):
    BUY = 1
    SELL = -1


@dataclass
class Trade:
    instrument_info: InstrumentInfo
    direction: TradeDirection
    price: float  # price per instrument (not per lot)
    quantity: int  # number of lots
    time: datetime.datetime  # local time


####################################################################################################
# Last Price
####################################################################################################

@dataclass
class LastPrice:
    instrument_info: InstrumentInfo
    price: float  # price per instrument (not per lot)
    time: datetime.datetime  # local time


####################################################################################################
# Candles
####################################################################################################

class CandleInterval(Enum):
    MIN_1 = 1
    MIN_5 = 5


@dataclass
class Candle:
    instrument_info: InstrumentInfo
    interval: CandleInterval
    # prices per instrument (not per lot)
    open: float
    high: float
    low: float
    close: float
    volume: int  # number of lots
    time: datetime.datetime  # begin of candle, local time
    last_trade_time: datetime.datetime  # last trade in candle, local time


####################################################################################################
# Stream Logger
####################################################################################################

class StreamLatencyLogger:
    """
    For managing stream message time differences and strategy latency
    """

    def __init__(self, stream_name: str, logger: logging.Logger) -> None:
        self._stream_name = stream_name
        self._logger = logger
        self._start_time: datetime.datetime | None = None
        self._subscribe_time: datetime.datetime | None = None
        self._last_message_time: datetime.datetime | None = None

    def on_start_listening(self) -> None:
        """
        Is called before receiving messages from streams
        """
        self._start_time = utc_now()

    def on_success_subscribe(self) -> None:
        assert self._start_time is not None
        assert self._subscribe_time is None, 'Got two subscription info messages'
        self._subscribe_time = utc_now()
        self._last_message_time = self._subscribe_time
        self._logger.info(f'{self._stream_name} subscribe time diff: {self._diff_ms(self._start_time, self._subscribe_time):.2f}ms')

    def on_recv_message(self, exchange_time_utc: datetime.datetime) -> None:
        """
        Is called after constructing event from stream, before strategy is called
        """
        assert exchange_time_utc.tzinfo == datetime.timezone.utc
        curr_time = utc_now()
        self._logger.debug(f'{self._stream_name} difference between two messages: {self._diff_ms(self._last_message_time, curr_time):.2f}ms')
        self._logger.debug(f'{self._stream_name} difference between message recv and exchange time: {self._diff_ms(exchange_time_utc, curr_time):.2f}ms')
        self._last_message_time = curr_time

    def on_strategy_done(self) -> None:
        """
        Is called after strategy has processed event
        """
        curr_time = utc_now()
        self._logger.debug(f'{self._stream_name} strategy callback time diff: {self._diff_ms(self._last_message_time, curr_time):.3f}ms')

    def is_ready(self) -> bool:
        """
        Return True <=> on_success_subscribe() is called
        """
        return self._subscribe_time is not None

    @staticmethod
    def _diff_ms(start_s: datetime.datetime, finish_s: datetime.datetime) -> float:
        return (finish_s - start_s).total_seconds() * 1e3


####################################################################################################
# MarketManager (main logic)
####################################################################################################

class MarketManager:
    def __init__(self):
        self._logger = Logging.get_logger('MarketManager')

        self._strategy: tp.Union['Strategy', None] = None
        self._rn: tp.Union['Runner', None] = None
        self._im: InfoManager | None = None
        self._services: AsyncServices | None = None
        self._market_data_stream: AsyncMarketDataStreamManager | None = None

        self._is_stream_ready: dict[str, bool] = {}
        self._notify_strategy = False

        self._latency_loggers: dict[str, StreamLatencyLogger] = {}

    ####################################################################################################
    # Methods for Strategy
    ####################################################################################################

    def subscribe_order_book(self, instruments: list[InstrumentInfo], depth: int) -> None:
        """
        Subscribe for order_book with given depth (1, 10, 20, 30, 40, 50)
        """
        self._logger.info(f'subscribe for order_book with {depth=} for {instruments}')
        assert depth in [1, 10, 20, 30, 40, 50], f'depth {depth} is not supported'  # Sanity check
        # add subscription request
        self._market_data_stream.order_book.subscribe(
            [inv.OrderBookInstrument(figi=instrument.figi, depth=depth) for instrument in instruments]
        )
        # create latency logger for streams
        for instrument in instruments:
            self._on_stream_creation(f'order_book_{depth}_depth_{instrument.figi}')

    def subscribe_trades(self, instruments: list[InstrumentInfo]) -> None:
        """
        Subscribe for last trades for given instruments
        Also download all last hour trades for the instrument
        """
        self._logger.info(f'subscribe for trades for {instruments}')
        # add subscription request
        self._market_data_stream.trades.subscribe(
            [inv.TradeInstrument(figi=instrument.figi) for instrument in instruments]
        )
        # create latency logger for streams
        for instrument in instruments:
            self._on_stream_creation(f'trades_{instrument.figi}')

    def subscribe_last_prices(self, instruments: list[InstrumentInfo]) -> None:
        """
        Subscribe for last prices for given instruments
        """
        self._logger.info(f'subscribe for last_prices for {instruments}')
        # add subscription request
        self._market_data_stream.last_price.subscribe(
            [inv.LastPriceInstrument(figi=instrument.figi) for instrument in instruments]
        )
        # create latency logger for streams
        for instrument in instruments:
            self._on_stream_creation(f'last_price_{instrument.figi}')

    def subscribe_candles(self, instruments: list[InstrumentInfo], interval: CandleInterval, from_: datetime.datetime | None = None):
        """
        Subscribe for candle-stream and download history candles
        Download history candles starting with from_ time
        Possible interval: 1 min, 5 min, 15 min, 1 hour, 1 day
        """
        self._logger.info(f'subscribe for candles ({interval=}, {from_=}) for {instruments}')
        # add subscription request
        self._market_data_stream.candles.subscribe(
            [inv.CandleInstrument(figi=instrument.figi, interval=self._local_interval_to_tinkoff(interval)) for instrument in instruments]
        )
        # create latency logger for streams
        for instrument in instruments:
            self._on_stream_creation(f'candles_{interval.value}_{instrument.figi}')

    ####################################################################################################
    # Methods for Runner
    ####################################################################################################

    def set_services(self, services: AsyncServices):
        """
        Only for use in Runner (not in strategy)
        """
        self._services = services
        self._market_data_stream = self._services.create_market_data_stream()

    def set_strategy(self, strategy: 'Strategy'):
        """
        Only for use in Runner (not in strategy)
        """
        self._strategy = strategy

    def set_helpers(self, rn: 'Runner', im: InfoManager):
        """
        Only for use in Runner (not in strategy)
        """
        self._rn = rn
        self._im = im

    async def run(self):
        """
        MarketManager loop of events
        """
        self._logger.info('START RUN')
        for latency_logger in self._latency_loggers.values():
            latency_logger.on_start_listening()  # start listening streams
        if self._check_readiness():
            # in case there are no streams
            return
        async for market_data in self._market_data_stream:
            market_data: inv.MarketDataResponse
            self._logger.debug(f'Got from stream: {market_data}')
            if market_data.subscribe_order_book_response is not None:
                self._process_subscribe_order_book(market_data.subscribe_order_book_response)
            if market_data.subscribe_trades_response is not None:
                self._process_subscribe_trades(market_data.subscribe_trades_response)
            if market_data.subscribe_last_price_response is not None:
                self._process_subscribe_last_price(market_data.subscribe_last_price_response)
            if market_data.subscribe_candles_response is not None:
                self._process_subscribe_candles(market_data.subscribe_candles_response)
            if market_data.orderbook is not None:
                self._process_order_book(market_data.orderbook)
            if market_data.trade is not None:
                self._process_trade(market_data.trade)
            if market_data.last_price is not None:
                self._process_last_price(market_data.last_price)
            if market_data.candle is not None:
                self._process_candle(market_data.candle)
            if all(item is None for item in [
                market_data.subscribe_order_book_response,
                market_data.subscribe_trades_response,
                market_data.subscribe_last_price_response,
                market_data.subscribe_candles_response,
                market_data.orderbook,
                market_data.trade,
                market_data.last_price,
                market_data.candle
            ]):
                # message is not processed
                self._logger.error(f'UNKNOWN RESPONSE: {market_data}')

    def start_strategy(self):
        """
        Is called when both MarketManager and UserManager are ready
        Start notifying strategy
        """
        self._notify_strategy = True

    ####################################################################################################
    # Private Methods for processing subscription messages
    ####################################################################################################

    def _process_subscribe_order_book(self, response: inv.SubscribeOrderBookResponse):
        for subscription in response.order_book_subscriptions:
            assert subscription.subscription_status == inv.SubscriptionStatus.SUBSCRIPTION_STATUS_SUCCESS, f'{subscription}'
            self._logger.info(f'subscribe for {subscription.figi} success: {subscription}')
            self._on_stream_success_subscribe(f'order_book_{subscription.depth}_depth_{subscription.figi}')

    def _process_subscribe_trades(self, response: inv.SubscribeTradesResponse):
        for subscription in response.trade_subscriptions:
            assert subscription.subscription_status == inv.SubscriptionStatus.SUBSCRIPTION_STATUS_SUCCESS, f'{subscription}'
            self._logger.info(f'subscribe for {subscription.figi} success: {subscription}')
            self._on_stream_success_subscribe(f'trades_{subscription.figi}')

    def _process_subscribe_last_price(self, response: inv.SubscribeLastPriceResponse):
        for subscription in response.last_price_subscriptions:
            assert subscription.subscription_status == inv.SubscriptionStatus.SUBSCRIPTION_STATUS_SUCCESS, f'{subscription}'
            self._logger.info(f'subscribe for {subscription.figi} success: {subscription}')
            self._on_stream_success_subscribe(f'last_price_{subscription.figi}')

    def _process_subscribe_candles(self, response: inv.SubscribeCandlesResponse):
        for subscription in response.candles_subscriptions:
            assert subscription.subscription_status == inv.SubscriptionStatus.SUBSCRIPTION_STATUS_SUCCESS, f'{subscription}'
            self._logger.info(f'subscribe for {subscription.figi} success: {subscription}')
            self._on_stream_success_subscribe(f'candles_{self._tinkoff_interval_to_local(subscription.interval).value}_{subscription.figi}')

    ####################################################################################################
    # Private Methods for processing updates
    ####################################################################################################

    ####################################################################################################
    # OrderBook
    ####################################################################################################

    def _process_order_book(self, order_book: inv.OrderBook):
        if not order_book.is_consistent:
            self._logger.warning(f'OrderBook is not consistent: {order_book}')
        order_book_data = OrderBook(
            instrument_info=self._im.get_instrument_by_figi(order_book.figi),
            depth=order_book.depth,
            bids=self._parse_order_book_levels(order_book.bids),
            asks=self._parse_order_book_levels(order_book.asks),
            time=utc_to_local(order_book.time),
            limit_up=quotation_to_float(order_book.limit_up),
            limit_down=quotation_to_float(order_book.limit_down)
        )
        latency_logger = self._latency_loggers[f'order_book_{order_book.depth}_depth_{order_book.figi}']
        latency_logger.on_recv_message(order_book.time)
        if self._notify_strategy:
            self._strategy.on_order_book_update(order_book_data)
            latency_logger.on_strategy_done()

    @staticmethod
    def _parse_order_book_levels(levels: list[inv.Order]) -> list[Order]:
        return [Order(price=quotation_to_float(level.price), quantity=level.quantity) for level in levels]

    ####################################################################################################
    # Trade
    ####################################################################################################

    def _process_trade(self, trade: inv.Trade):
        trade_data = Trade(
            instrument_info=self._im.get_instrument_by_figi(trade.figi),
            direction=self._parse_trade_direction(trade.direction),
            price=quotation_to_float(trade.price),
            quantity=trade.quantity,
            time=utc_to_local(trade.time),
        )
        latency_logger = self._latency_loggers[f'trades_{trade.figi}']
        latency_logger.on_recv_message(trade.time)
        if self._notify_strategy:
            self._strategy.on_market_trade(trade_data)
            latency_logger.on_strategy_done()

    @staticmethod
    def _parse_trade_direction(direction: inv.TradeDirection) -> TradeDirection:
        match direction:
            case inv.TradeDirection.TRADE_DIRECTION_BUY:
                return TradeDirection.BUY
            case inv.TradeDirection.TRADE_DIRECTION_SELL:
                return TradeDirection.SELL
            case inv.TradeDirection.TRADE_DIRECTION_UNSPECIFIED:
                raise RuntimeError(f'Unknown direction in trade: {direction}')
            case _:
                assert False, 'Unreachable'

    ####################################################################################################
    # LastPrice
    ####################################################################################################

    def _process_last_price(self, last_price: inv.LastPrice):
        last_price_data = LastPrice(
            instrument_info=self._im.get_instrument_by_figi(last_price.figi),
            price=quotation_to_float(last_price.price),
            time=utc_to_local(last_price.time)
        )
        latency_logger = self._latency_loggers[f'last_price_{last_price.figi}']
        latency_logger.on_recv_message(last_price.time)
        if self._notify_strategy:
            self._strategy.on_last_price(last_price_data)
            latency_logger.on_strategy_done()

    ####################################################################################################
    # Candles
    ####################################################################################################

    def _process_candle(self, candle: inv.Candle):
        last_price_data = Candle(
            instrument_info=self._im.get_instrument_by_figi(candle.figi),
            interval=self._tinkoff_interval_to_local(candle.interval),
            open=quotation_to_float(candle.open),
            high=quotation_to_float(candle.high),
            low=quotation_to_float(candle.low),
            close=quotation_to_float(candle.close),
            volume=candle.volume,
            time=utc_to_local(candle.time),
            last_trade_time=utc_to_local(candle.last_trade_ts)
        )
        latency_logger = self._latency_loggers[f'candles_{last_price_data.interval.value}_{candle.figi}']
        latency_logger.on_recv_message(candle.last_trade_ts)
        if self._notify_strategy:
            self._strategy.on_candle(last_price_data)
            latency_logger.on_strategy_done()

    @staticmethod
    def _tinkoff_interval_to_local(candle_interval: inv.SubscriptionInterval) -> CandleInterval:
        match candle_interval:
            case inv.SubscriptionInterval.SUBSCRIPTION_INTERVAL_ONE_MINUTE:
                return CandleInterval.MIN_1
            case inv.SubscriptionInterval.SUBSCRIPTION_INTERVAL_FIVE_MINUTES:
                return CandleInterval.MIN_5
            case _:
                assert False, 'Unreachable'

    @staticmethod
    def _local_interval_to_tinkoff(candle_interval: CandleInterval) -> inv.SubscriptionInterval:
        match candle_interval:
            case CandleInterval.MIN_1:
                return inv.SubscriptionInterval.SUBSCRIPTION_INTERVAL_ONE_MINUTE
            case CandleInterval.MIN_5:
                return inv.SubscriptionInterval.SUBSCRIPTION_INTERVAL_FIVE_MINUTES
            case _:
                assert False, 'Unreachable'

    ####################################################################################################
    # Private Methods for logging latency and checking Manager's readiness
    ####################################################################################################

    def _on_stream_creation(self, stream_name: str):
        """
        Is called (once per stream) in subscribe methods for creating a stream
        """
        self._latency_loggers[stream_name] = StreamLatencyLogger(stream_name, self._logger)
        self._is_stream_ready[stream_name] = False

    def _on_stream_success_subscribe(self, stream_name: str):
        """
        Is called (once per stream) when received a success subscription message
        """
        self._latency_loggers[stream_name].on_success_subscribe()
        self._is_stream_ready[stream_name] = True
        self._check_readiness()  # notify runner if all subscriptions are done

    def _check_readiness(self) -> bool:
        """
        Check readiness of all subscriptions and history market data
        Tell Runner about readiness if needed
        """
        all_subscriptions_are_ready = all(latency_logger.is_ready() for latency_logger in self._latency_loggers.values())
        if all_subscriptions_are_ready:
            self._rn.on_market_manager_ready()
            return True
        return False
