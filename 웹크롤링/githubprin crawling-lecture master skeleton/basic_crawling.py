import requests
from bs4 import BeautifulSoup

def crawl_breaking_news_list():
    news_url = 'https://news.naver.com/main/list.naver?mode=LPOD&mid=sec&sid1=001&sid2=140&oid=001&isYeonhapFlash=Y&aid=0014907888'

    response = requests.get(news_url)

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        
        td = soup.find('td', {'class' : 'content'})
        
        for li in td.find_all('li'):
            try:
                if li['data-comment'] is not None:
                    a = li.find('a')
                    link = a['href']
                    text = a.text
                    print(link, text)
            except KeyError:
                pass
        

def crawl_ranking_news():
    ranking_url = 'https://news.naver.com/main/ranking/popularDay.naver'
    
    response = requests.get(ranking_url)

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        
        td = soup.find('div', {'class' : 'rankingnews _popularWelBase _persist'})
        for rankingnews_box in td.find_all('div', {'class' : 'rankingnews_box'}):
            # <rankingnews_box ....> 
            # <div class = 'rankingnews_box' .. >
            
            strong = rankingnews_box.find('strong')
            print('\n\n언론사 이름:', strong.text)
            
            for li in rankingnews_box.find_all('li'):
                try:
                    if li is not None:
                        b = li.find('div', {'class' : 'list_content'})
                        if b is not None:
                            a = b.find('a')
                        
                            text = a.text
                            link = a['href']
                            print('\t - ', text,'\n\t\t', link)
                        else: pass
                except KeyError:
                    pass

''' 강사님 ver
def crawl_ranking_news():
    ranking_url = 'https://news.naver.com/main/ranking/popularDay.naver'

    response = requests.get(ranking_url)

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')

        for div in soup.find_all('div', {'class' : 'rankingnews_box'}):
            press_name = div.find('strong')
            print('===============')
            print(press_name.text)

            for li in div.find_all('li'):
                content = li.find('div', {'class': 'list_content'})
                if content is not None:
                    a = content.find('a') 
                    print(a['href'], a.text)
'''

if __name__ == '__main__':
    crawl_breaking_news_list()
    crawl_ranking_news()
    
    # 언론사 \n 제목  링크