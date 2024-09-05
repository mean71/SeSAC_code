import asyncio
import aiohttp
import requests
from bs4 import BeautifulSoup
from time import time

def crawl_press_names_and_codes():
    
    pass

# ????









async def afetch_journalist_list(press_code, session):
    url = f'https://media.naver.com/journalists/whole?officeId={press_code}'
    
    response = await session.get(url)
    
    if response.status_code ==200:
        # do something here
        print('good!')
    
    await response.release()

async def acrawl_journalist_info_pages(code2name):
    session = aiohttp.ClientSession()
    
    tasks = [afetch_journalist_list(press_code, session) for press_code in code2name]
    
    results = await asyncio.gather(*tasks)
    