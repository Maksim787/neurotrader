import logging

from pathlib import Path
from dataclasses import dataclass


@dataclass
class LoggerInfo:
    name: str
    logger: logging.Logger | None = None

    stdout_log_level: int | None = None
    stdout_handler: logging.Handler | None = None

    file_log_level: int | None = None
    file_handler: logging.Handler | None = None

    def check_correctness(self) -> None:
        assert all(map(lambda x: x is not None, [self.logger, self.stdout_log_level, self.stdout_handler, self.file_log_level, self.file_handler])), 'Sanity check'


class Logging:
    """
    Singleton utility for managing loggers
    """

    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL
    levels = [DEBUG, INFO, WARNING, ERROR, CRITICAL]

    # common formatter
    formatter = logging.Formatter(
        fmt='[%(asctime)s.%(msecs)03d] [%(levelname)s] [%(name)s] {%(module)s} - %(funcName)s(): %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # LoggerInfo by logger name
    loggers: dict[str, LoggerInfo] = {}

    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """
        Construct or get logger by name
        """
        return cls._construct_logger(name).logger

    @classmethod
    def set_file_log_level(cls, name: str, level: int):
        """
        Change log level for writing in file
        """
        cls._set_log_level(name, level, is_file=True)

    @classmethod
    def set_stdout_log_level(cls, name: str, level: int):
        """
        Change log level for writing in stdout
        """
        cls._set_log_level(name, level, is_file=False)

    @classmethod
    def _construct_logger(cls, name: str) -> LoggerInfo:
        """
        Construct logger with given name
        """
        # check existence
        if name in cls.loggers:
            return cls.loggers[name]

        # create directory
        log_directory = Path('connector/logs')
        if not log_directory.exists():
            log_directory.mkdir(exist_ok=True)

        # create LoggerInfo
        cls.loggers[name] = LoggerInfo(name=name)
        logger_info = cls.loggers[name]

        # stdout handler
        stdout_handler = logging.StreamHandler()
        stdout_handler.setFormatter(cls.formatter)
        stdout_handler.setLevel(logger_info.stdout_log_level or cls.DEBUG)
        logger_info.stdout_log_level = stdout_handler.level
        logger_info.stdout_handler = stdout_handler

        # file handler
        file_handler = logging.FileHandler(
            filename=log_directory / f"{name}.log",
            mode="w",
            encoding="utf-8"
        )
        file_handler.setFormatter(cls.formatter)
        file_handler.setLevel(logger_info.file_log_level or cls.DEBUG)
        logger_info.file_log_level = file_handler.level
        logger_info.file_handler = file_handler

        # logger
        logger = logging.Logger(
            name=name,
            level=logging.DEBUG  # do not change logger level, change handlers' levels
        )
        logger.addHandler(stdout_handler)
        logger.addHandler(file_handler)
        logger_info.logger = logger
        logger_info.check_correctness()  # Sanity check
        return logger_info

    @classmethod
    def _set_log_level(cls, name: str, level: int, is_file: bool) -> None:
        """
        Change log level for writing in stdout or in file
        """
        assert level in cls.levels, 'logging level not found'
        if name not in cls.loggers:
            cls.loggers[name] = LoggerInfo(name=name)
        logger_info = cls.loggers[name]
        if is_file:
            logger_info.file_log_level = level
            if logger_info.file_handler is not None:
                logger_info.file_handler.setLevel(level)
        else:
            logger_info.stdout_log_level = level
            if logger_info.stdout_handler is not None:
                logger_info.stdout_handler.setLevel(level)
