import random

class Game: 
    def __init__(self, userDB, player_id): # 딕셔너리 userDB, id를 키값으로 받아서
        self.userDB = userDB
        self.player_id = player_id
        self.initial_rating = userDB[player_id]['initial_rating']
        self.actual_rating = userDB[player_id]['actual_rating']
        self.win_lose_history = userDB[player_id]['win_lose_history']
        self.match_players

    def play_match(self):
        # player1, player2의 win_lose_history를 update하고
        # elo rating 알고리즘에 따라 각자의 current_rating을 update할 것
        # https://namu.wiki/w/Elo%20%EB%A0%88%EC%9D%B4%ED%8C%85 참고
        player2_id = self.matach_players()
        W = self.simulate(other)
        K=30    # 점수가중치
        self.userDB[id]['initial_rating'] +=  K*(W - We) # self 점수 변동
        other.userDB[id]['initial_rating'] += K*(other.W - We) # other 점수 변동
        
        if W == 1:
           self.win_lose_history['win']+=1
           other.win_lose_history['lose']+=1
        elif W == 0:
           self.win_lose_history['lose']+=1
           other.win_lose_history['win']+=1
        elif W == 0.5:
           self.win_lose_history['draw']+=1
           other.win_lose_history['draw']+=1
        # 예측승률We = 1 / 10**( (상대initial_rating - 유저initial_rating)/400 + 1 )
        # 경기결과W = 예측승률을 끌어와 무작위 분배, 승리=1, 무승부=0.5, 패배=0
        # 경기후점수Pa = 경기전점수Pb + 가중치K*(경기결과W - 예측승률We)
        # K 20, We 50%: 승리자 +10점, 패배자 -10점, 무승부 변동 없음
        # K 20, We 75%: 강자승리 강자+5,약자-5; 무승부 강자-5,약자+5; 약자승리 강자-15,약자+15
        # 나와 상대의 티어점수 주소를 호출해서 가감하고 집어넣어야 한다.
        # 데이터부터 짜고 생성
        pass

    def match_players(self): #인자로 받은 userDB에서 키값ID리스트를 뽑아 상대로 랜덤매칭
        player_lst= list(userDB.keys())
        player_lst.remove(self.player_id)
        return random.choice(player_lst) # 유저리스트에서 랜덤으로 하나 매칭

    def simulate(self,other): # match_players 불러와서 둘이 매칭
        We = 1 / (10**( (other.initial_rating - self.initial_rating)/400 ) + 1)  # self.userDB[id]의 레이팅 기대승률
        Wes = 1 / (10**( (other.actual_rating - self.actual_rating)/400 ) + 1)
        W = random.choices([1, 0.5, 0], weights=[Wes, 0.01-Wes*0.01,1-Wes], k=1)[0] #유저레이팅 예측승률기반 가중치로 승패결과반환
        return W
        # We를 기반으로한 가상 대전결과 # 대전결과W = 승리1, 무승부0.5, 패배0

class Player:
    def __init__(self, userDB, id): # id,게임티어,유저별 실제 실력티어
        self.win_lose_history = {
            'win'=0,
            'lose'=0,
            'draw'=0
        }
        self.current_rating = initial_rating
        self.actual_rating = actual_rating

    def __str__(self):
        return userDB[id]


userDB ={} # 유저DB
user_name = set() # 중복방지

def mk_userDB(new_id, old_id): # 중복없이 신규/기존 테스트 유저DB 랜덤생성횟수 지정
  anystr = 'abcdefghijklmnopqrstuvwxyz123456789'
  mk_user = new_id + old_id
  while len(user_name) < mk_user:
    id = ''.join(random.choices(anystr, k=10))
    if id not in user_name:
      user_name.add(id)
      actual_rating = random.randint(700, 5000)
      if len(user_name) <= new_id:
        initial_rating = 1000
      else:
        initial_rating = random.randint(2000, 7000)
    userDB[id] = {
        'actual_rating':actual_rating,
        'initial_rating': initial_rating,
        'win_lose_history': {
            'win' : 0,
            'lose' : 0,
            'draw' : 0
            }
        }




if __name__ == '__main__':

    mk_userDB(5,10) # 유저DB랜덤생성(신규유저수,숙련유저수)
    print(len(user_name),[*user_name]) # 확인용
    print(len(userDB),userDB,sep='\n') # 생성DB출력

    player1_id = random.choice(list(userDB.keys()))
    player1 = Player(userDB, player1_id)

    game = Game(userDB,player1_id)






#일단 name 티어 실력 만들고
#딕셔너리에 집어넣고
#userDB[user_name][]
# id 셋이 아니라 id 딕셔너리를 만들고 그곳에 레이팅과 유저레이팅을 딕셔너리로 입력 받으려면?
# id생성을 닉네임으로 바꾸고 클래스 인스턴스id를 생성해서 클래스메서드로 id를 1부터 부여한다면

# 그것을 호출하려면?
# 유저데이터딕셔너리에 매번 생성되는 유저 'id':{rating:점수, user_rating:실력점수} 2차원? 딕셔너리 부여
# user_data_dic[id][rating] = 1000
# user_data_dic[id][user_rating] = 1234

# user_data_dic[id][rating] = 1000
# user_data_dic[id][user_rating] = 1234

# player 데이터는 랜덤함수로 몇백개생성 해서 SQL파일에 넣을 수 있을까? 안보고는 불가능
# id 는 그냥 일정범위 숫자와 문자를 랜덤생성
# elo rating알고리즘에 따라 레이팅차이별 승률 설정

# 코드 실행 결과
# id : , 레이팅 : , 실력레이팅 :
# 을 전부 출력해서 결과를 볼 수 있도록 한다. 근데 모든유저끼리 대전시켜서 시뮬?
# 정신건강을 위해 한유저만 







# 데코레이터 함수란
def my_decorator(func):
    def wrapper(*args, **kwargs):
        # 함수 호출 전 코드
        result = func(*args, **kwargs)
        # 함수 호출 후 코드
        return result
    return wrapper





기본 난수 생성 함수들:

random.random(): 0.0과 1.0 사이의 임의의 부동 소수점 숫자를 반환합니다.
random.uniform(a, b): a와 b 사이의 임의의 부동 소수점 숫자를 반환합니다.
random.randint(a, b): a와 b 사이의 임의의 정수를 반환합니다.
random.randrange(start, stop[, step]): 주어진 범위 내에서 임의의 정수를 반환합니다.
random.choice(seq): 주어진 시퀀스에서 임의의 요소를 반환합니다.
random.choices(population, weights=None, *, cum_weights=None, k=1): 주어진 모집단에서 가중치를 고려하여 중복 허용 k개의 요소를 반환합니다.
random.sample(population, k): 모집단에서 k개의 고유한 요소를 무작위로 선택하여 반환합니다.
random.shuffle(x[, random]): 시퀀스를 제자리에서 무작위로 섞습니다.

분포 함수
random.betavariate(alpha, beta): 베타 분포 난수를 반환합니다.
random.expovariate(lambd): 지수 분포 난수를 반환합니다.

# 백준 실5 배열정렬
N = int(input())
words,word = [],[]
for i in range(N):
    words.append(input())


words = list(set(words))
words.sort(key=lambda x: (len(x),x))
print(*words, sep = '\n')


from datetime import datetime
today = datetime.today()
print(str(today)[0:10])

집합의 내장함수
    
