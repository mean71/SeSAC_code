import asyncio # 비동기 프로그래밍을 가능하게 하는 모듈
import time # 코드 실행 시간을 측정하기 위해 time 모듈을 임포트
# 비동기 함수 정의 (async def)
async def async_task(task_number, duration):
    print(f'started {task_number}') # 비동기 작업이 시작되었음을 출력
    await asyncio.sleep(duration) # time.sleep과 비슷하다. 비동기성 수면 ?# asyncio.sleep을 사용하여 await가 될때까지 비동기적으로 대기 (duration만큼 대기)
    print(f'ended {task_number}') # 비동기 작업이 끝났음을 출력

async def main(): # async함수는 일단 실행되면 무조건 반환부터한다 실행 끝나면 넘겨준다는 약속을 반환
    task1 = asyncio.create_task( async_task(1, 3) ) # task_number=1, duration=3초
    task2 = asyncio.create_task( async_task(2, 2) ) # task_number=2, duration=2초
    task3 = asyncio.create_task( async_task(3, 1) ) # task_number=3, duration=1초
    # 각각의 비동기 작업이 완료될 때까지 대기
    await task1
    await task2
    await task3

# 두 번째 메인 함수 정의 (강사님 피셜 더 깔끔한 문법)
async def gather_main():
    # asyncio.gather를 사용하여 비동기 작업을 동시에 실행
    await asyncio.gather(
        async_task(1, 3),
        async_task(2, 2),
        async_task(3, 1)
    )
# 코드 실행 시간 측정을 위해 시작 시간 기록
begin = time.time()
asyncio.run(main()) # 비동기 함수 실행 (main 함수 호출)
end = time.time() # 코드 실행이 끝난 후 종료 시간 기록

print(end - begin)
# 웹크롤링은 각자 가져오는시간이 제각각이라 비동기프로그래밍으로 짜야 시간과 자원을 절약한다.