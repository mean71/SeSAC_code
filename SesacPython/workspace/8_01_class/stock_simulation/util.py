import random
import string
import matplotlib.pyplot as plt

def generate_random_word(length): # 길이를 입력받아 for문에 순환, 자음/모음 번갈아서 랜덤으로 합쳐 첫글자는 .capitatlize로 출력
    vowels = 'aeiou'
    consonants = "".join(set(string.ascii_lowercase) - set(vowels)) # 자음문자열을 join으로 합쳐서 대입 # string.ascii_lowercase #set()으로 문자열의 순서를 없애고 나눔 #string내부의 소문자 알파벳을 저장해둔 문자열 - 모음문자열
    word = ""
    
    for i in range(length):  
        if i % 2 == 0:
            word += random.choice(consonants)
        else:
            word += random.choice(vowels)
    
    return word.capitalize() # 앞글자를 대문자로?하는 .capitalize()적용한 문자열로 반환

def generate_company_name():  # 상장사명 반환 #접두사+접미사+단어뽑기 로 랜덤생성
    prefix_length = random.randint(2, 4)  # 2~4사이 랜덤정수대입
    suffix_length = random.randint(3, 5)  # 3~5사이
    
    prefix = generate_random_word(prefix_length)  #  2~4길이랜덤단어 접두사
    suffix = generate_random_word(suffix_length)  #  3~5길이랜덤단어 접미사
    
    suffixes = ["Tech", "Corp", "Solutions", "Systems", "Industries", "Enterprises", "Dynamics", "Holdings", "Ventures", "Innovations"]
    suffix_choice = random.choice(suffixes) # 단어리스트에서 랜덤하게 선택
    
    company_name = f"{prefix}{suffix} {suffix_choice}" 
    return company_name

def company_lists(n):  # 입력값 만큼 랜덤주식회사 리스트를 생성하여 반환
    res = []
    while len(res) < n:
        cand = generate_company_name()
        if cand not in res:
            res.append(cand)
    return res 

def plot_line_graph(data, save_to = 'sample.png', title="Line Graph", x_label="X-axis", y_label="Y-axis"):        
    plt.figure(figsize=(10, 6))
    plt.plot(data, marker='o', linestyle='-', color='b')
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True)
    plt.savefig(save_to)
    plt.close()
# MATLAB플로팅라이브러리를 이용한 데이터시각화함수
if __name__ == '__main__': # 실행시 데이터값 100개를 매트랩라이브러리로 그래프로 표현
    plot_line_graph(range(100)) 