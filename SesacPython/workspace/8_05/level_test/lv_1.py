all_smallcase_letters = 'abcdefghijklmnopqrstuvwxyz'
# --------------------------------------------
# 1. list/tuple 기초 예제들 
# a는 1,2,3이 들어간 튜플, 
# b는 a부터 z까지 모든 알파벳 소문자가 들어간 리스트가 되도록 만들어보세요. 
# b를 만들 때 위에 주어진 all_smallcase_letters와 for loop를 사용해도 좋고, 손으로 다 쳐도 좋습니다. 
# --------------------------------------------
a,b = (1,2,3),[]
b.extend(all_smallcase_letters)
print('1',type(b), b)
# --------------------------------------------
# 2. dict 기초 예제 
# 1) upper_to_lower
# upper_to_lower은 모든 대문자 알파벳(ex. A)을 key로 가지고, 대응하는 소문자 알파벳(ex. a)을 value로 가지는 dict입니다. 
# 위에서 만든 b와 for loop를 이용해서 upper_to_lower을 만들어보세요.
upper_to_lower, lower_to_upper, alpha_to_number, number_to_alphabet = {},{},{},{}

B=[i.upper() for i in b]
print('2_list',type(B), B)
for x, y in zip(B,b):
  upper_to_lower[x]=y
print('2_1',type(upper_to_lower), upper_to_lower)
# 2) lower_to_upper
# upper_to_lower과 반대로 된 dict를 만들어보세요. 
for x, y in zip(b,B):
  upper_to_lower[x]=y
print('2_2',type(upper_to_lower), upper_to_lower)
# 3) alpha_to_number
# 소문자 알파벳 각각을 key, 몇 번째 알파벳인지를 value로 가지는 dict를 만들어보세요. 
# 위 all_smallcase_letters와 enumerate함수를 사용하세요. 
# 알파벳 순서는 1부터 시작합니다. ex) alpha_to_number['a'] = 1
for x,y in enumerate(all_smallcase_letters):
  alpha_to_number[y]=x+1
print('3',type(alpha_to_number), alpha_to_number)
# 4) number_to_alphabet
# 1부터 26까지의 수를 key로, 소문자, 대문자로 이뤄진 문자열 2개의 튜플을 value로 가지는 dict를 만들어보세요.
for x,y in enumerate(tuple(zip(b,B))):
  number_to_alphabet[x+1]=y
print('4',type(number_to_alphabet), number_to_alphabet)
# --------------------------------------------
# 3. 주어진 문자열의 대소문자 바꾸기 
# 위 2에서 만든 dict들을 이용하여, 아래 문제들을 풀어보세요.
A = 'absdf123SAFDSDF'
# 1) 주어진 문자열을 모두 대문자로 바꾼 문자열을 만들어보세요.
A_upper=[]
for a in A:
  if 'a'<= a <= 'z':
    A_upper.append(a.swapcase())
  else:
    A_upper.append(a)
A_upper=''.join(A_upper)
print('31',A_upper)
# 2) 주어진 문자열을 모두 소문자로 바꾼 문자열을 만들어보세요.
A_lower=[]
for a in A:
  if 'A'<= a <= 'Z':
    A_lower.append(a.swapcase())
  else:
    A_lower.append(a)
A_lower=''.join(A_lower)
print('32',A_lower)
# 3) 주어진 문자열에서 대문자는 모두 소문자로, 소문자는 모두 대문자로 바꾼 문자열을 만들어보세요.
A_swapcase = A.swapcase()
print('32',A_swapcase)
# 4) 주어진 문자열이 모두 알파벳이면 True, 아니면 False를 출력하는 코드를 짜보세요.
print('34 True') if A.isalpha() else print('34 False')
# --------------------------------------------
# 4. 다양한 패턴 찍어보기 
# 1) 피라미드 찍어보기 - 1 
# 다음 패턴을 프린트해보세요.
#     *
#    ***
#   *****
#  *******
# *********
# --------------------------------------------
def pyramid1(N):
    for n in range(N):
        print(f'{' '*(N-n)}{"*"*(2*n+1)}')
    pass
pyramid1(5)
# --------------------------------------------
# 2) 피라미드 찍어보기 - 2 
# 다음 패턴을 프린트해보세요.
#     * 
#    * * 
#   * * * 
#  * * * * 
# * * * * * 
# --------------------------------------------
def pyramid2(n):
    for n in range(1,N+1):
        print( ' '*(N-n), end='' )
        print( '* '*n)
    pass
pyramid2(5)
# --------------------------------------------
# 3) 피라미드 찍어보기 - 3
# 다음 패턴을 프린트해보세요.
#     A 
#    A B 
#   A B C 
#  A B C D 
# A B C D E 
# --------------------------------------------
def pyramid3(n):
    x="ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for n in range(1, N+1):
        print( ' '*(N-n), end='' )
        for i in range(n):
            if i > 25:
                print( f'{x[i-26]} ', end = '' ) 
            else:
                print( f'{x[i]} ', end = '' )
        print()
    pass
pyramid3(5)
# --------------------------------------------
# 4) 피라미드 찍어보기 - 4
# 다음 패턴을 프린트해보세요. 
#       1 
#      1 1 
#     1 2 1 
#    1 3 3 1 
#   1 4 6 4 1
# --------------------------------------------
N = int(input())
for i in range(N):
  for j in range(i+1):
    
# --------------------------------------------
# 5) 다음 패턴을 찍어보세요. 
# *         *         * 9 6
#   *       *       *   7 5
#     *     *     *     5 4
#       *   *   *       3 3
#         * * *         1 2
#           *           
#         * * *         
#       *   *   *       
#     *     *     *     
#   *       *       *   
# *         *         * 
# --------------------------------------------
print('*         *         *')
def mk_str(n):
  if n < 1:
     return None
  elif n != 1:
    for i in range(2,n+1):
      print( ' '*(n*2-i) + '*' + ' '*(i*2-3) + '*' + ' '(i*2-3) + '*' + ' '*(n*2-i) )
  print('  '*n+'*'+'  '*n)

N = int(input())
mk_str(N)