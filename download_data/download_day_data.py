import asyncio
import aiohttp
import aiomoex
import pandas as pd
from pathlib import Path


async def download_ticker_task(session: aiohttp.ClientSession, data_folder: Path, security: str):
    print(f'Download {security}')
    data = await aiomoex.get_market_history(session, security)
    security_df = pd.DataFrame(data)
    file_path = data_folder / f'{security}.csv'
    security_df.to_csv(file_path, index=False)
    print(f'Success: {security}')


async def main():
    data_folder = Path('data/day_data/')
    data_folder.mkdir(exist_ok=True)

    async with aiohttp.ClientSession() as session:
        data = await aiomoex.get_board_securities(session)
        all_securities_df = pd.DataFrame(data)
        securities = all_securities_df['SECID']
        print(f'securities: {securities}')
        await asyncio.gather(*[download_ticker_task(session, data_folder, security) for security in securities])


asyncio.run(main())
