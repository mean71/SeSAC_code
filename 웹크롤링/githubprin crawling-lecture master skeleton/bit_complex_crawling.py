import requests 
from bs4 import BeautifulSoup
# pip install beautifulsoup4
def crawl_press_names_and_codes():
    """Make the dict that have press code as key, and press name as value. Crawl from https://media.naver.com/channel/settings."""
    url = 'https://media.naver.com/channel/settings'
    response = requests.get(url)
    
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        
        td = soup.find('div', {'class' : 'rankingnews _popularWelBase _persist'})
    
    
    code2name = { 이름 : 코드}
    return code2name 

def crawl_journalist_info_pages(code2name):
    """Accepts press code - press name dict, and return dict having press code as key,  and 2-tuple of (press name, listof 2-tuple containing journalist name and their link) as value. 

    For now, you DO NOT have to crawl all journalists; for now, it's impossible. 
    Crawl from https://media.naver.com/journalists/. 
    """

    res = {}
    
    return res

class Journalist:
    def __init__(self, name, press_code,
                career_list,
                graduated_from,
                no_of_subscribers,
                subscriber_age_statistics,
                subscriber_gender_statistics,
                article_list):
        self.name = name
        self.press_code = press_code 
        self.career_list = career_list
        self.graduated_from = graduated_from
        self.no_of_subscribers = no_of_subscribers
        self.subscriber_age_statistics = subscriber_age_statistics
        self.subscriber_gender_statistics = subscriber_gender_statistics
        self.article_list = article_list 

def crawl_journalist_info(link):
    """Make a Journalist class instance using the information in the given link. 
    """
    pass 
    
if __name__ == '__main__':
    code2name = crawl_press_names_and_codes()
    code2info = crawl_journalist_info_pages(code2name)
    for code, (press_name, journalist_list) in code2info.items():
        for journalist_name, link in journalist_list:
            j = crawl_journalist_info(link)
            assert j.name = journalist_name

    
    