# 14_2d_loop.py
# 학습 날짜 : 8월 28일 수요일
# 파이썬 프리코스 8회차
# 학습 내용 : 중첩 반복문


# y는 0 부터 9까지 증가한다.
for y in range(0, 10):
    # x는 0 부터 9까지 증가한다.

    for x in range(0, 10):
        # y와 x를 동시에 출력, 값의 변환를 관찰
        print(f"y = {y}, x = {x}")
        """
        y = 0, x = 0
        y = 0, x = 1
        y = 0, x = 2
        y = 0, x = 3
        y = 0, x = 4
        y = 0, x = 5
        y = 0, x = 6
        y = 0, x = 7
        y = 0, x = 8
        y = 0, x = 9
        y = 1, x = 0
        y = 1, x = 1
        y = 1, x = 2
        y = 1, x = 3
        ...
        """

# 이차원리스트 : 리스트를 저장한 리스트
# 일차원리스트 : 선의 모양
list_1 = [
    [0, 1, 2, 4],  # index = 0
    [3, 4],  # index = 1
    [6],  # index = 2
    [9, 10, 11, 3, 7],  # index = 3
]

# 1중 반복문(바깥 리스트의 반복문)
list_1_length = len(list_1)
for index in range(list_1_length):
    print(index, list_1[index])
    in_list = list_1[index]
    in_list_length = len(in_list)
    # 2중 반복문(내부 리스트의 반복문)
    for in_index in range(in_list_length):
        print(in_index, in_list[in_index])

    """
    0 [0, 1, 2]
    1 [3, 4, 5]
    2 [6, 7, 8]
    3 [9, 10, 11]
    """


# 이차원리스트에 대한 2중 반복문
# 리스트들을 감싸는 바깥 리스트에 대한 반복문 1개
# 바깥 리스트 내부에 있는 각 리스트들에 대한 반복문 1개

list_2 = [
    [0],
    [1, 2],
    [3, 4, 5],
]

# 바깥 리스트를 반복하는 반복문
list_2_len = len(list_2)
for y in range(list_2_len):
    # 바깥 리스트에 대한 인덱싱 코드
    in_list = list_2[y]

    in_list_len = len(in_list)
    for x in range(in_list_len):
        # 내부 리스트에 대한 인덱싱 코드
        number = in_list[x]
        print(number)


list_2 = [
    [0],
    [1, 2],
    [3, 4, 5],
]
print(list_2[0][0])
print(list_2[1][0], list_2[1][1])
print(list_2[2][0], list_2[2][1], list_2[2][2])

list_3 = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
]
total = 0
# list_3에 저장된 숫자들의 합을 구하는 코드를 작성.

# 내부 리스트의 개수
list_3_len = len(list_3)

for y in range(list_3_len):
    # 내부의 각 리스트
    in_list = list_3[y]

    in_list_len = len(in_list)
    for x in range(in_list_len):
        number = in_list[x]
        total = total + number

print(f"total = {total}")


# 딕셔너리에 대한 반복문
# 리스트에 대한 반복문 -> 내부 원소의 반복
# 딕셔너리에 대한 반복문 -> 내부 원소의 반복
# 딕셔너리는 원소에 key - value 2개의 데이터가 있다.

dict_1 = {
    "일": 1,
    "이": 2,
    "삼": 3,
}

# 딕셔너리의 반복문은 키(key)를 기준으로 반복한다.
for key in dict_1:
    # key가 있기 때문에 value에 대해 인덱싱
    value = dict_1[key]
    print(key, value)
