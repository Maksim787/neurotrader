import aiohttp
import asyncio
import aiofiles
from bs4 import BeautifulSoup
from json import loads
import re
import os
import requests


REQUIRED_LIMIT = 10000


def load_data(security: str,
              engine: str = 'stock',
              market: str = 'shares',
              interval: int = 1,
              start_date: str = '2017-01-01',
              end_date: str = '2023-01-01',
              filename=None,
              sep=' '):
    """
    Loading securities candles from iss.moex.com/iss

    :param security: Name of security
    :param engine: Type of engine (need for url)
    :param market: Type of market (need for url)
    :param interval: Duration of each candle (1, 10, ...)
    :param start_date: Start of period in format yyyy-mm-dd
    :param end_date: End of period in format yyyy-mm-dd
    :param filename: File to write candles (engine-market-security-interval by default)
    :param sep: Separator in one candle
    :return: None
    """
    if filename is None:
        filename = f'data/{engine}-{market}-{security}-{interval}.txt'
        with open(filename, 'w') as file:
            start: int = 0
            while True:
                url = f'https://iss.moex.com/iss/engines/{engine}/markets/{market}/securities/{security}/candles.json?start={start}&interval={interval}&from={start_date}&till={end_date}'
                soup = BeautifulSoup(requests.get(url).text, features="lxml")
                data = loads(soup.body.p.text)['candles']['data']
                if len(data) == 0:
                    break
                for row in data:
                    str_row = sep.join([str(el) for el in row])
                    if re.sub(r'[0-9]+.', '', str_row):
                        continue
                    file.write(str_row + '\n')
                start += len(data)


async def async_load_data(session: aiohttp.ClientSession,
                          security: str,
                          engine: str = 'stock',
                          market: str = 'shares',
                          interval: int = 1,
                          start_date: str = '2017-01-01',
                          end_date: str = '2023-01-01',
                          filename=None,
                          sep=' '):
    """
    Concurrent loading securities candles from iss.moex.com/iss (aiohttp instead of requests)

    :param session: Common parameter for all functions in program
    :param security: Name of security
    :param engine: Type of engine (need for url)
    :param market: Type of market (need for url)
    :param interval: Duration of each candle (1, 10, ...)
    :param start_date: Start of period in format yyyy-mm-dd
    :param end_date: End of period in format yyyy-mm-dd
    :param filename: File to write candles (engine-market-security-interval by default)
    :param sep: Separator in one candle
    :return: None
    """
    if filename is None:
        filename = f'data/{engine}-{market}-{security}-{interval}.txt'

    start: int = 0
    try:
        async with aiofiles.open(filename, 'w') as file:
            while True:
                url = f'https://iss.moex.com/iss/engines/{engine}/markets/{market}/securities/{security}/candles.json?start={start}&interval={interval}&from={start_date}&till={end_date}'
                async with session.get(url) as response:
                    soup = BeautifulSoup(await response.text(), features="lxml")
                    data = loads(soup.body.p.text)['candles']['data']
                    if len(data) == 0:
                        break
                    result_row = ""
                    for row in data:
                        str_row = sep.join([str(el) for el in row])
                        if re.sub(r'[0-9]+.', '', str_row):
                            continue
                        result_row += str_row + '\n'
                    await file.write(result_row)
                    start += len(data)
    except Exception as err:
        # Maybe this security need to be loaded again
        print(f'Failed for {security} with {err}')
    finally:
        print(f'Ended for {security} with {start} rows')
        # Removing files with small history
        if start < REQUIRED_LIMIT:
            os.remove(filename)


async def async_gather(names: list[str], **kwargs):
    """
    Common runner for all async loaders

    :param names: Names of securities - one for each loader
    :param kwargs: Key-word arguments for async loaders
    :return: None
    """
    async with aiohttp.ClientSession() as session:
        await asyncio.gather(*[async_load_data(session, security=name, **kwargs) for name in names])


def make_securities(filename: str = 'securities.txt'):
    """
    Loading names of securities from security statistics

    :param filename: Path to output file
    :return: None
    """
    url = f'https://iss.moex.com/iss/engines/stock/markets/shares/secstats'
    soup = BeautifulSoup(requests.get(url).text, features="lxml")
    names = list(set(el['secid'] for el in soup.body.document.data.rows.find_all('row')))
    final_names = []
    for name in sorted(names):
        try:
            url = f'https://iss.moex.com/iss/engines/stock/markets/shares/securities/{name}/candles.json?start=0&interval=1&from=2020-01-01&till=2023-01-01'
            soup = BeautifulSoup(requests.get(url).text, features="lxml")
            data = loads(soup.body.p.text)['candles']['data']
            if len(data) == 0:
                continue
            final_names.append(name)
        except Exception:
            pass

    with open(filename, 'w') as sec:
        sec.writelines([name + '\n' for name in final_names])


if __name__ == '__main__':
    if not os.path.exists('securities.txt'):
        make_securities()

    with open('securities.txt', 'r') as sec:
        names = [name.strip() for name in sec.readlines()]

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        coroutine = async_gather(names=names, interval=10)
        loop.run_until_complete(coroutine)
        loop.close()
    except KeyboardInterrupt:
        pass
