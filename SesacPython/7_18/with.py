simple_python = '''
a = 1
b = 2
a+b+3
'''

def eval_simple_python(text):
    list_text = text.strip().split("\n")
    namespace = {}
    res = None
    for line in list_text:
        if line != '':
            res, namespace = eval_line(line, namespace)
    return res

def eval_line(line, namespace):
  if is_assingment(line):
    name, value = parse_assignment(line)
    namespace[name] = value
  elif is_expression(line):
    value = eval_expression(line)
  return value, namespace

def is_assingment(line):
  if '=' in line:
    left, right = line.split('=')
    name = left.strip() # check if name is valid
    expr =  right.strip()
    return True
  else: return False

def is_expression(line):
  return not is_assingment(line)

def parse_assignment(line, namespace):
    left, right = line.split("=")
    name = left.strip() # check if name is valid
    expr = right.strip()
    value = eval_expression(expr, namespace)
    return name, value

def eval_expression(line, namespace):
    values = map(lambda v:v.strip(), line.split('+'))
    result = 0
    for v in values:
        if v.isnumeric():
            result += int(v)
        else:
            if v in namespace:
                result += namespace[v]
            else:
                raise NameError("")
    return result


with open('text.txt', 'r') as file:
    content = file.read()
    print('print : \n',content)
    result = eval_simple_python('text.txt')
    print('result:', result)

with open('text.txt', 'w') as file:
    file.write()
    # 예외에러발생시키기
    # 마지막줄이 (assignment)아니면,계산값리턴 맞으면 None 리턴 계산 불가하면 NameError 리턴


def process_text(content):
    try:
        with open('text.txt', 'w') as file:
            file.write(content)
        last_line = content.strip().split('\n')[-1]
        if "assignment" in last_line:
            return None
        else:
            try:
                return eval(last_line)  # 계산식이 가능할 경우 리턴
            except NameError:
                raise NameError("계산 불가: NameError 발생")  # 계산 불가 시 NameError 발생   
    except Exception as e:
        # 예외가 발생할 경우 예외 메시지 출력
        print(f"예외 발생: {e}")
        return None

# 사용 예시
content = "3 + 5\nassignment"
result = process_text(content)
print(result)  # None

content = "3 + 5\n2 * 6"
result = process_text(content)
print(result)  # 12

content = "3 + 5\nundefined_variable"
result = process_text(content)
# NameError