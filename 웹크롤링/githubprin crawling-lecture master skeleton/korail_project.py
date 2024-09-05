import asyncio
# import playwright
from playwright.async_api import async_playwright
from time import time, sleep

# import os 
# print(playwright.__file__) # 패키지실행이 안되서 경로추척

# 하나부터 열까지 직접 짜는 korail예매 매크로
# 라이브러리와 html 숙련도가 부족하여 이번엔 시간을 아끼기위해 큰 틀은 대충 프롬프트 엔지니어링
async def main():
    async with async_playwright() as p:
        # 브라우저 실행 (GUI 모드)
        browser = await p.chromium.launch(headless=False);aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa=0;id='';pw='';pww=''
        context = await browser.new_context()
        # 새 페이지 열기
        page = await context.new_page()
        
        # 코레일 홈페이지로 이동
        await page.goto('https://www.letskorail.com')
        # 페이지가 완전히 로드될 때까지 대기
        await page.wait_for_load_state('networkidle')  # 'networkidle', 'load', 'domcontentloaded'
        # 예매 버튼에 마우스를 올리고 클릭 버튼의 이미지요소 우클릭후 셀렉터 요소를 복붙
        await page.click('#res_cont_tab01 > form > div > fieldset > p > a > img')# 예매버튼 : 이미지 요소를 클릭
        # 클릭 후 페이지의 추가 로딩이 완료될 때까지 대기
        await page.wait_for_load_state('networkidle')
        
        # await korail_input(input_keyword = '부산') # 출발역 : '서울'
        # await korail_input(input_keyword = '서울') # 도착역 : '부산'
        
        # 출발역 입력칸 선택하고 입력
        await page.fill('#start', '서울')  # 출발역 셀렉터 필요
        # 도착역 입력칸 선택하고 입력
        await page.fill('#get', '부산')  # 도착역 셀렉터 필요
        # 날짜 및 시간 선택
        await page.select_option('#s_year', '2024')  # 연도 선택 셀렉터
        await page.select_option('#s_month', '9')  # 월 선택 셀렉터
        await page.select_option('#s_day', '9')  # 일 선택 셀렉터
        await page.select_option('#s_hour', '9 (오전09)')  # 최소 출발 시간 선택
        # 하단 '조회하기' 버튼 클릭 후 로딩 대기
        await page.click('#center > div.ticket_box > p > a > img')  # 조회 버튼 셀렉터대입
        await page.wait_for_load_state('networkidle')  # 페이지 로딩 완료 대기
        
        # 모든 예매 버튼을 찾기
        buttons = await page.query_selector_all('tr a[href^="javascript:infochk("]')
        # 가장 먼저 나오는 예매 버튼 클릭
        if buttons:
            first_button = buttons[0]
            await first_button.click()
            print("가장 먼저 나오는 예매 버튼을 클릭했습니다.")
        else:
            print("예매 버튼을 찾을 수 없습니다.")
        await page.wait_for_load_state('networkidle')  # 페이지 로딩 완료 대기
        
        # 로그인
        # await page.wait_for_selector('#txtMember')  # ID 입력 필드 기다리기 # 페이지 로딩이 끝나도 동적으로 생성되거나 비동기적으로 추가되는 요소는 렌더링 되었다는 보장없음
        await page.fill('#txtMember', id)  # id = 'id'
        await page.fill('#txtPwd', pw)  # pw = 'qw'
        await page.click('#loginDisplay1 > ul > li.btn_login > a > img')  # 조회 버튼 셀렉터대입
        await page.wait_for_load_state('networkidle')  # 페이지 로딩 완료 대기
        # 결제버튼
        await page.click('#btn_next > span')  # 결제하기 버튼
        await page.wait_for_load_state('networkidle')  # 페이지 로딩 완료 대기
        await page.click('#chk_smpl_sb') # 등록한 계좌로
        await page.click('#fnIssuing')
        await page.wait_for_load_state('networkidle')  # 페이지 로딩 완료 대기
        # 이후 결제 팝업 창 대기
        popup = await context.wait_for_event('page')
        await popup.wait_for_load_state('networkidle')
        

        # 팝업 창에서 필요한 작업 수행
        await popup.click('body > div.popup_ly > div.sp_cont > div.my_card > div.bank_cd1 > a')  # 적절한 셀렉터로 대체
        await page.wait_for_load_state('networkidle')  # 페이지 로딩 완료 대기
        await popup.click('#inputPwd1')  # 입력 필드를 클릭하여 활성화
        # 직접 비번 입력 (숫자 버튼 클릭)
        for char in pww:
            await popup.keyboard.press(char)
        # 결제 확인
        await popup.click('body > div.sp_bottom > a > p')
        # 페이지가 로드된 후 추가 작업 수행
        print("페이지 로드 완료. 창을 수동으로 닫으세요.")
        # 대기 (이 부분은 테스트 후 주석 처리 가능)
        await asyncio.sleep(120)  # 60초 동안 대기하여 브라우저 창이 닫히지 않도록 합니다.

if __name__=='__main__':
    asyncio.run(main())