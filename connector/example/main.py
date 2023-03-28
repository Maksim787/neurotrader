import asyncio
import yaml

from connector.runner import Runner
from connector.example.strategy import MyStrategy
from connector.common.log import Logging


async def main() -> None:
    logger.info('Create Runner and Strategy')
    # Read token
    with open('connector/private/keys.yaml') as f:
        private_keys = yaml.safe_load(f)
        token = private_keys['token']
        account_id = str(private_keys['account_id'])
    # Read config
    with open('connector/example/config.yaml') as f:
        config = yaml.safe_load(f)

    ticker = config['ticker']

    # Create strategy
    strategy = MyStrategy(ticker=ticker)
    logger.info('Run Strategy')
    async with Runner(token, account_id) as runner:
        await runner.run(strategy)  # main loop


if __name__ == '__main__':
    Logging.set_log_directory('connector/logs/')
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
