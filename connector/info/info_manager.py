from dataclasses import dataclass

from tinkoff.invest.async_services import AsyncServices
from connector.common.log import Logging


@dataclass
class InstrumentInfo:
    ticker: str  # strategy instrument id
    class_code: str
    figi: str  # tinkoff-investment instrument id
    instrument_type: str  # etf/share/...


class InfoManager:
    def __init__(self):
        self._logger = Logging.get_logger('InfoManager')
        self._services: AsyncServices | None = None
        self._instrument_info_by_ticker: dict[str, InstrumentInfo] = {}  # InstrumentInfo by ticker
        self._instrument_info_by_figi: dict[str, InstrumentInfo] = {}  # InstrumentInfo by figi

    def set_services(self, services: AsyncServices):
        self._services = services

    async def download_instruments_info(self):
        """
        Download instrument info for all instruments in favourites
        """
        self._logger.info('Download favourite instruments')
        favourites = await self._services.instruments.get_favorites()
        for instrument in favourites.favorite_instruments:
            instrument_info = InstrumentInfo(ticker=instrument.ticker, class_code=instrument.class_code, figi=instrument.figi, instrument_type=instrument.instrument_type)
            self._logger.debug(f'Found instrument: {instrument_info}')
            self._instrument_info_by_figi[instrument_info.figi] = instrument_info
            self._instrument_info_by_ticker[instrument_info.ticker] = instrument_info

    def get_instrument_by_ticker(self, ticker: str) -> InstrumentInfo:
        return self._instrument_info_by_ticker[ticker]

    def get_instrument_by_figi(self, figi: str) -> InstrumentInfo:
        return self._instrument_info_by_figi[figi]

    def get_all_instruments(self) -> list[InstrumentInfo]:
        return list(self._instrument_info_by_figi.values())
