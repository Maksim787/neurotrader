import asyncio
from dataclasses import dataclass

import tinkoff.invest as inv
from tinkoff.invest.async_services import AsyncServices
from connector.common.log import Logging
from connector.common.quotation import quotation_to_float


@dataclass
class InstrumentInfo:
    figi: str  # tinkoff-investment instrument id
    price_increment: float
    ticker: str
    class_code: str
    instrument_type: str  # etf/share/...
    name: str  # instrument name
    trade_available_flag: bool


class InfoManager:
    def __init__(self):
        self._logger = Logging.get_logger('InfoManager')
        self._services: AsyncServices | None = None
        self._instrument_info_by_ticker: dict[str, InstrumentInfo] = {}  # InstrumentInfo by ticker
        self._instrument_info_by_figi: dict[str, InstrumentInfo] = {}  # InstrumentInfo by figi

    ####################################################################################################
    # Methods for Runner
    ####################################################################################################

    def set_services(self, services: AsyncServices):
        self._services = services

    async def download_instruments_info(self):
        """
        Download instrument info for all instruments in favourites
        """
        self._logger.info('Download favourite instruments')
        favourites = (await self._services.instruments.get_favorites()).favorite_instruments
        self._logger.debug(f'Favourite instruments: {favourites}')
        instrument_infos = await asyncio.gather(*[self.download_instrument_info(instrument) for instrument in favourites])
        self._instrument_info_by_figi = {instrument_info.figi: instrument_info for instrument_info in instrument_infos}
        self._instrument_info_by_ticker = {instrument_info.ticker: instrument_info for instrument_info in instrument_infos}

    ####################################################################################################
    # Private Methods
    ####################################################################################################

    async def download_instrument_info(self, favorite_instrument: inv.FavoriteInstrument) -> InstrumentInfo:
        instrument: inv.Etf | inv.Share
        match favorite_instrument.instrument_type:
            case 'etf':
                method_name = 'etf_by'
                instrument = (await self._services.instruments.etf_by(id_type=inv.InstrumentIdType.INSTRUMENT_ID_TYPE_FIGI, id=favorite_instrument.figi)).instrument
            case 'share':
                method_name = 'share_by'
            case _:
                raise RuntimeError('Unsupported instrument_type')
        method = getattr(self._services.instruments, method_name)
        instrument = (await method(id_type=inv.InstrumentIdType.INSTRUMENT_ID_TYPE_FIGI, id=favorite_instrument.figi)).instrument
        return InstrumentInfo(
            figi=favorite_instrument.figi,
            price_increment=quotation_to_float(instrument.min_price_increment),
            ticker=favorite_instrument.ticker,
            class_code=favorite_instrument.class_code,
            instrument_type=favorite_instrument.instrument_type,
            name=instrument.name,
            trade_available_flag=instrument.buy_available_flag and instrument.sell_available_flag and not instrument.for_qual_investor_flag
        )

    ####################################################################################################
    # Methods for Strategy and other Managers
    ####################################################################################################

    def get_instrument_by_ticker(self, ticker: str) -> InstrumentInfo:
        return self._instrument_info_by_ticker[ticker]

    def get_instrument_by_figi(self, figi: str) -> InstrumentInfo:
        return self._instrument_info_by_figi[figi]

    def get_all_instruments(self) -> list[InstrumentInfo]:
        return list(self._instrument_info_by_figi.values())

    def has_instrument_by_ticker(self, ticker: str) -> bool:
        return ticker in self._instrument_info_by_ticker

    def has_instrument_by_figi(self, figi: str) -> bool:
        return figi in self._instrument_info_by_figi
