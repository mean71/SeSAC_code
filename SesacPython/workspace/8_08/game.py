import random

class Game: 
    def __init__(self, userDB, player_id): # 딕셔너리 userDB, id를 키값으로 받아서
        self.userDB = userDB
        self.player_id = player_id
        self.initial_rating = userDB[player_id]['initial_rating']
        self.actual_rating = userDB[player_id]['actual_rating']
        self.win_lose_history = userDB[player_id]['win_lose_history']
        self.player2_id = self.match_players()

    def play_match(self):
        W = self.simulate()
        player2_win_lose_history = userDB[self.player2_id]['win_lose_history']
        if W == 1:
           self.win_lose_history['win']+=1
           player2_win_lose_history['lose']+=1
        elif W == 0:
           self.win_lose_history['lose']+=1
           player2_win_lose_history['win']+=1
        print(f'{Player(self.userDB,self.player_id)}')
        print(f'{Player(self.userDB,self.player2_id)}')
        # 예측승률We = 1 / 10**( (상대initial_rating - 유저initial_rating)/400 + 1 )
        # 경기결과W = 예측승률을 끌어와 무작위 분배, 승리=1, 무승부=0.5, 패배=0
        # 경기후점수Pa = 경기전점수Pb + 가중치K*(경기결과W - 예측승률We)
        # K 20, We 50%: 승리자 +10점, 패배자 -10점, 무승부 변동 없음
        # K 20, We 75%: 강자승리 강자+5,약자-5; 무승부 강자-5,약자+5; 약자승리 강자-15,약자+15
        # 나와 상대의 티어점수 주소를 호출해서 가감하고 집어넣어야 한다.
        # 데이터부터 짜고 생성
    def match_players(self): #인자로 받은 userDB에서 키값ID리스트를 뽑아 상대로 랜덤매칭
        player_lst= list(userDB.keys())
        player_lst.remove(self.player_id)
        return random.choice(player_lst) # 매칭시킬 유저리스트에서 랜덤으로 다른 id하나 찾아서 반환

    def simulate(self): # match_players 불러와서 둘이 매칭
        player2_id = self.player2_id
        player2_initial_rating = userDB[player2_id]['initial_rating']
        player2_actual_rating = userDB[player2_id]['actual_rating']
        
        We = 1 / (10**( (player2_initial_rating - self.initial_rating)/400 ) + 1)  # self.userDB[id]의 레이팅 기대승률
        actual_rating_We = 1 / (10**( (player2_actual_rating - self.actual_rating)/400 ) + 1) # 실제실력스탯에 기반한 예측승률
        W = random.choices([1, 0], weights=[actual_rating_We, 1-actual_rating_We], k=1)[0]
         #유저레이팅 예측승률기반 가중치로 승패결과반환
        K=100    # 점수가중치
        self.userDB[self.player_id]['initial_rating'] +=  K*(W - We) # self_player 점수 변동
        if W == 1:
            self.userDB[self.player2_id]['initial_rating'] += K*(0 - (1-We)) # player2 점수 변동
        elif W == 0:
            self.userDB[self.player2_id]['initial_rating'] += K*(1 - (1-We)) # player2 점수 변동
        return W
        # We를 기반으로한 가상 대전결과 # 대전결과W = 승리1, 무승부0.5, 패배0

class Player:
    def __init__(self, userDB, player_id): # id,게임티어,유저별 실제 실력티어
        self.userDB = userDB
        self.player_id = player_id
        # self.win_lose_history = {
        #     'win':0,
        #     'lose':0,
        # }
        # self.current_rating = initial_rating
        # self.actual_rating = actual_rating
        
    def __str__(self):
        user = self.userDB[self.player_id]
        return f'{user['initial_rating']}{user['actual_rating']}'


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
            'lose' : 0
            }
        }


if __name__ == '__main__':

    mk_userDB(5,10) # 유저DB랜덤생성(신규유저수,숙련유저수)
    print(len(user_name),[*user_name]) # 확인용
    print(len(userDB),userDB,sep='\n') # 생성DB출력
    a = 0
    while a < 200000:
        player1_id = random.choice(list(userDB.keys()))
        game = Game(userDB,player1_id)
        game.play_match()
        a+=1
    for x,b in userDB.items():
        print(f'ID:{x} _ initial_rating:{b['initial_rating']} _ actual_rating:{b['actual_rating']} _ {b['win_lose_history']}', sep='\n')





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
# 정신건강을 위해 한유저만 둘다 랜덤으로! 증강점수 출력?