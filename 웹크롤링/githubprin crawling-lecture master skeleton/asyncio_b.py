# pip install aiohttp

import asyncio
import aiohttp

async def afetch(url, session): # session 매번만들면 낭비니 넣어줌
    response = await session.get(url)

    # do something with response
    if response.status_code == 200:
        return await response.release() # do something here


async def main():
    urls = [...]
    # session = aiohttp.ClientSession()
    
    tasks = [afetch_and_process(url, session) for url in urls]
    
    results = await asyncio.gather(*tasks)
    
    await session.close()

if __name__=='__main__':
    asyncio.run(main()) 