import typing as tp

from tinkoff.invest.async_services import AsyncServices
from connector.info.info_manager import InfoManager

if tp.TYPE_CHECKING:
    from connector.runner import Runner


class UserManager:
    def __init__(self):
        self._strategy = None
        self._rn: tp.Union['Runner', None] = None
        self._im: InfoManager | None = None
        self._services: AsyncServices | None = None

    def set_services(self, services: AsyncServices):
        """
        Only for use in Runner (not in strategy)
        """
        self._services = services

    def set_strategy(self, strategy):
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
        UserManager loop of events
        """
        self._rn.on_user_manager_ready()
