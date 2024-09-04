class MyDate:

    def __init__(self, year = 0, month = 0, day = 0, hour = 0, minute = 0, sec = 0): #__init__(self, ...): 생성자 메서드로, 객체가 생성될 때 호출됩니다. 객체의 초기 상태를 설정하는 데 사용됩니다.
        self.year = year
        self.month = month
        self.day = day
        self.hour = hour
        self.minute = minute
        self.sec = sec
        self.check_DB()
        print(f'{year:04d}-{month:02d}-{day:02d}-{hour:02d}:{minute:02d}:{sec:02d}')

    def all_date(self):
        return (self.year, self.month, self.day, self.hour, self.minute, self.sec)

    def check_DB(self): # 범위지정
        assert not 0 <= self.year, f'RangeErrorYear ({self.year})'
        assert not 0 <= self.month <= 12, f'RangeErrorMonth ({self.month})'
        assert not 0 <= self.day <= self.count_month_day(self.year, self.month), f'RangeErrorDay ({self.day})'
        assert not 0 <= self.hour < 24, f'RangeErrorHour ({self.hour})'
        assert not 0 <= self.minute < 60, f'RangeErrorMinute ({self.minute})'
        assert not 0 <= self.sec < 60, f'RangeErrorSec ({self.sec})'
        print('good')

    def what_y(self, year):
        return (year % 4 == 0 and year % 100 != 0) or year % 400 == 0

    def count_month_day(self, year, month):
        if month == 2:
            return 29 if self.what_y(year) else 28
        return [31, 0, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][month - 1]

    @classmethod
    def cm(cls): pass
    @staticmethod
    def sm(data): pass
    
   
    def __add__(self, other): # a + b는 a.__add__(b)        # a+b #a는 기준 시간이고, b는 그n년뒤n달뒤n일뒤n시n분의 세월로 간주하고 따로 추가해본다.
        year = self.year + other.year
        month = self.month + other.month
        day = self.day + other.day
        hour = self.hour + other.hour
        minute = self.day + other.minute
        print('__add__')
        print(year, month, day, hour, minute)
        if month > 12:
            month %= 12
            year += 1
        if minute > 59:
            minute %= 60
            day += 1
        if hour > 23:
            hour %= 24
            day += 1
        if day > self.count_month_day(year, month):
            day %= self.count_month_day(year, month)
            month += 1
            if month > 12:
                month %= 12
                year += 1
        
        
        return MyDate(year, month, day, hour, minute)


    def __sub__(self, other): # a - b는 a.__sub__(b) 
        year = self.year - other.year
        month = self.month - other.month
        day = self.day - other.day
        hour = self.hour - other.hour
        minute = self.minute - other.minute

        if month <= 0:
            month += 12
            year -= 1
        if minute < 0:
            minute += 60
            hour -= 1
        if hour < 0:
            hour += 24
            day -= 1
        if day < 0:
            day += self.count_month_day(year, month)
            month -= 1
            if month <= 0:
                month += 12
                year -= 1
        print('__sub__')
        print(year, month, day, hour, minute)
        return MyDate(year, month, day, hour, minute)


    def __eq__(self, other): # a == b는 a.__eq__(b)        # a==b
        return MyDate.all_date(self) == MyDate.all_date(other)

    def __lt__(self, other): # a < b는 a.__lt__(b)        # a<b
        return MyDate.all_date(self) < MyDate.all_date(other)
    
    def __le__(self, other): # a <= b는 a.__le__(b)        # a<=b
        return MyDate.all_date(self) <= MyDate.all_date(other)

    def __gt__(self, other): # a > b는 a.__gt__(b)        # a>b
        return MyDate.all_date(self) > MyDate.all_date(other)

    def __ge__(self, other): # a >= b는 a.__ge__(b)        # a>=b
        return MyDate.all_date(self) >= MyDate.all_date(other)
    
    # def __str__(self): # 유사메서드 __repr__# 이러면 작동하나?
    #      return f"{self.year}-{self.month}-{self.day} {self.hour:02d}:{self.minute:02d}:{self.sec:02d}"

if __name__ == '__main__':
    d0 = MyDate()
    d1 = MyDate(2022, 4, 1, 14, 30) # 일단 시:분 까지만

    d2 = MyDate(2024, 8, 100, 23, 10) # should raise an error
    d3 = MyDate(2024, 2, 30)

    d3 = MyDate(day = 1) 
    assert d1 + d3 == MyDate(2022, 4, 2, 14, 30), 'is not d1 + d3'
    assert d1 - d3 == MyDate(2022, 3, 31, 14, 30), f' {d1.all_date()}{d3.all_date()}, in not d1 - d3' # 코드는 잘돌아가지만 주소값이 나온다. 실수를 이해하기위해 보존
    assert d1 < d2, 'd1 < d2'
    assert d1 <= d2, 'd1 <= d2'
    assert d1 > d2, 'd1 > d2'
    assert d1 >= d2, 'd1 >= d2'

#def l():
 #   str(
    '''
    __init__(self, ...): 생성자 메서드로, 객체가 생성될 때 호출됩니다. 객체의 초기 상태를 설정하는 데 사용됩니다.
    __add__(self, other): 덧셈 연산자 +에 해당하며, 두 객체를 더할 때 호출됩니다.
    __sub__(self, other): 뺄셈 연산자 -에 해당하며, 두 객체를 뺄 때 호출됩니다.
    __eq__(self, other): 동등 연산자 ==에 해당하며, 두 객체가 같은지 비교할 때 호출됩니다.
    __lt__(self, other): 미만 연산자 <에 해당하며, 한 객체가 다른 객체보다 작은지 비교할 때 호출됩니다.
    __le__(self, other): 이하 연산자 <=에 해당하며, 한 객체가 다른 객체보다 작거나 같은지 비교할 때 호출됩니다.
    __gt__(self, other): 초과 연산자 >에 해당하며, 한 객체가 다른 객체보다 큰지 비교할 때 호출됩니다.
    __ge__(self, other): 이상 연산자 >=에 해당하며, 한 객체가 다른 객체보다 크거나 같은지 비교할 때 호출됩니다.

    d=Mydate(.)

    d = Mydate()
    => MyDate __ init __ (d, ...)
    => d = MyDate(year = 2024)


    __init__ (self, other,)
    self.year = year
    멤버변수들 초기화

    __add__
    Mydate 애들끼리 더할때

    __sub__
    Mydate 애들끼리 뺄때

    __eq__
    Mydate 애들이 같은지 test
    다른지 test하려면 (not __eq__ 부르면 될듯)

    __lt__
    ex) d1 < d2
    lt = less than

    __le__
    ex) d1 <= d2
    le = less equal

    __gt__
    ex) d1 > d2
    gt = greater than

    __ge__
    ex) d1 >= d2
    ge =greater equal



    ex) d1 + d2
    => MyDate __add__(d1, d2)

    ex) d1 - d2
    => MyDate __sub__(d1,d2)

    '''
    #)    pass