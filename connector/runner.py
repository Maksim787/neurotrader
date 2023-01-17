import asyncio

import tinkoff.invest as inv
from tinkoff.invest.async_services import AsyncServices
from connector.market.market_manager import MarketManager
from connector.user.user_manager import UserManager
from connector.info.info_manager import InfoManager
from connector.common.log import Logging


class Runner:
    """
    Run strategy
    Can only be used via 'async with'
    """

    def __init__(self, token: str):
        self._logger = Logging.get_logger('Runner')

        self._mm = MarketManager()
        self._um = UserManager()
        self._im = InfoManager()
        self._client = inv.AsyncClient(token)
        self._services: AsyncServices | None = None

        self._mm_ready = False
        self._um_ready = False

    ####################################################################################################
    # Methods for main.py
    ####################################################################################################

    async def __aenter__(self) -> 'Runner':
        """
        The only way to use Runner
        """
        self._services = await self._client.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self._client.__aexit__(exc_type, exc_val, exc_tb)

    async def run(self, strategy):
        """
        Start Managers and strategy, enter infinite loop
        """
        assert self._services is not None, 'Runner must be invoked using "async with"'

        # connect strategy with managers
        strategy.set_helpers(self._mm, self._um, self._im)

        # assign services
        self._um.set_services(self._services)
        self._mm.set_services(self._services)
        self._im.set_services(self._services)

        # assign strategy
        self._mm.set_strategy(strategy)
        self._um.set_strategy(strategy)

        # assign InfoManager
        self._mm.set_helpers(self, self._im)
        self._um.set_helpers(self, self._im)

        # download instruments (instrument_info is needed to start subscribing)
        await self._im.download_instruments_info()

        # subscribe for new data and download history data
        self._logger.info('Call strategy.subscribe()')
        strategy.subscribe()
        # TODO: mm and um subscription processing

        # history data is downloaded, all streams are open, call 'on_start'
        self._logger.info('Call strategy.on_start()')
        strategy.on_start()

        # run loops of MarketManager and UserManager
        self._logger.info('Start Managers')
        await asyncio.gather(self._mm.run(), self._um.run())

    ####################################################################################################
    # Methods for Managers
    ####################################################################################################

    def on_market_manager_ready(self):
        """
        Is called when all history data is downloaded and streams are active
        """
        self._mm_ready = True
        self._logger.info('MarketManager is ready')
        self._start_strategy()

    def on_user_manager_ready(self):
        """
        Is called when UserManager is ready to work
        """
        self._um_ready = True
        self._logger.info('UserManager is ready')
        self._start_strategy()

    ####################################################################################################
    # Private Methods
    ####################################################################################################

    def _start_strategy(self):
        """
        Start calling strategy callbacks from MarketManager if both Managers are ready
        """
        if self._mm_ready and self._um_ready:
            self._logger.info('Both managers are ready: start strategy')
            self._mm.start_strategy()
