#프로그래머스1번
# str = input()
# while True:
#     if 1 <= len(str) <= 1000000 and ' ' not in str:
#         print(str)
#         break
#     else:
#         continue

# while (str := input()) and 1<=len(str)<=1000000 and " " not in str:print(str);break

#프로그래머스2
#while(ab:=input().split(' '))and(all(AB/1==0)for AB in ab)and all(-100000<=len(AB)<=1000000 for AB in ab):print(f'a = {ab[0]}\nb = {ab[1]}');break

# 프로그래머스?
# while (ab := input().split(' ')) and (all(AB/1==0) for AB in ab) and all(-100000<=len(AB)<=1000000 for AB in ab):print(sum(map(int,ab)));break

#프로그래머스3
# str,n = input().split(' ');print(str*int(n))
#프로그래머스4
# while(str:=input())and 1<= len(str)<=20 and str.isalpha():print(str.swapcase());break
#프로그래머스5
#print(r'!@#$%^&*(\'"<>?:;')
#프로그래머스6
# while(ab:=list(map(int,input().strip().split(' '))))and ab[0]>=1 and ab[1]<=100:print(f'{ab[0]} + {ab[1]} = {ab[0]+ab[1]}');break
#프로그래머스7
# while(ab:=input().strip().split(' '))and all(1<=len(x)<=100 for x in ab):print(ab[0]+ab[1]);break
# print(input().strip().replace(' ', ''))
#프로그래머스8
#print('\n'.join(input()))
#print(*input(),sep='\n')
#while(str:=input())and 1<=len(str)<=10:[print(x)for x in str];break
#프로그래머스9
#while(n:=int(input()))and 1<=n<=1000:print(f'{n} is even')if n%2 ==0 else print(f'{n} is odd'); break
#print(f'{(n:=int(input()))} is {"odd" if n%2!=0 else "even"}')
#프로그래머스10
'''
import re

def solution(my_string, overwrite_string, s):
    pattern=r'^[0-9A-Za-z]+$'
    if re.match(pattern,my_string)and re.match(pattern,overwrite_string)and 1<=len(overwrite_string)<=len(my_string)<=1000 and 0<=s<=len(my_string)-len(overwrite_string):
        my_string = my_string[:s]+overwrite_string+my_string[s+len(overwrite_string) :]
    return my_string
'''

#프로그래머스11
# 길이가 같은 두 문자열 str1과 str2가 주어집니다.
# 두 문자열의 각 문자가 앞에서부터 서로 번갈아가면서 한 번씩 등장하는 문자열을 만들어 return 하는 solution 함수를 완성해 주세요.
# def solution(str1, str2):
#     answer=''
#     while True and(1<=len(str1)==len(str2)<=10)and str1==str1.lower()and str2==str2.lower():answer+=''.join(str1[i]+str2[i]for i in range(0,len(str1)));break
#     return answer
#프로그래머스12
# def solution(arr):
#     while not(answer := '') and 1<=len(arr)<=200 and (all(arr[x].islower()) for x in range(len(arr))):answer = ''.join(arr);break
#     return answer
#주사위 게임 1
'''
def solution(a, b):
    answer = 0
    if 1<=a<=6 and 1<=b<=6:
        if (a+b)%2==0:
            if a%2!=0: return a**2 + b**2
            else: return = abs(a-b)
        else: return = 2*(a+b)
'''
'''
def solution(a, b):
    if a%2 and b%2: return a*a+b*b
    elif a%2 or b%2: return 2*(a+b)
    return abs(a-b)'''
#주사위 게임 2
'''
def solution(a, b, c):
    answer = 0
    if (all(1<=x<=6) for x in (a,b,c)):
        if a!=b and a!=c and b!=c:
            return (a+b+c)
        elif  a!=b or b!=c or a!=c:
            return (a + b + c)*(a**2 + b**2 + c**2)
        else:
            return (a+b+c)*(a**2+b**2+c**2)*(a**3+b**3+c**3)
    return answer'''
'''
def solution(a, b, c):
    check=len(set([a,b,c]))
    if check==1:
        return 3*a*3*(a**2)*3*(a**3)
    elif check==2:
        return (a+b+c)*(a**2+b**2+c**2)
    else:
        return (a+b+c)'''
'''
def solution(a, b, c):
    answer=a+b+c
    if a==b or b==c or a==c: answer*=a**2+b**2+c**2
    if a==b==c:answer*=a**3+b**3+c**3
    return answer'''
#solution = lambda a, b, c: a+b+c if a!=b and b!=c and c!=a else ((a+b+c)*(a**2+b**2+c**2)*(a**3+b**3+c**3) if a==b and b==c else (a+b+c)*(a**2+b**2+c**2))
#주사위게임 3
'''

'''

# while(str:=list(input().strip().split(' ')))and all(1<=len(str[x])<=10)for x in range(len(str)) and
# t=(lambda s:(lambda:"Invalid" if s.replace('-', '', 1).isdigit() is False else (lambda x: "Positive" if x > 0 else "Negative" if x < 0 else "Zero")(int(s)))())(input("Enter an integer: "))

# result=(lambda s:(lambda x:"Positive" if x > 0 else "Negative" if x < 0 else "Zero")(int(s)) if s.replace('-', '', 1).isdigit() else "Invalid input")(input("Enter an integer: "))
