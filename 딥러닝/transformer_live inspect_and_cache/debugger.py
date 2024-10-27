import os
import sys
import linecache
# 파일의 특정 줄을 쉽게 읽어오는 기능 제공 -> linecache.getline(filename, lineno): filename의 lineno번째 줄을 읽어 반환
# linecache.clearcache(): 읽은 파일의 내용을 캐시에 저장하고, clearcache()는 이 캐시를 초기화

from code import interact, InteractiveInterpreter # interact(local=locals()) # interact() 실행 시점에서 local에 있는 변수를 확인하고 수정 가능
from pprint import pprint, pformat
'''
pprint
    pprint(data, width=, depth=, ) # 파이썬 3.8 옵션 -> sort_dicts=True  사전의 키를 정렬해서 출력
    pformat -> pretty print 결과를 문자열로 반환 : 이를 통해 파일로 저장하거나 다른곳에 출력하기전 사용가능
    전자는 pretty하게 출력 후자는 전자에서 출력될문자열을 반환만 한다.
'''

def debug_shell(line_window = 5):
    """
    Print the usual traceback information, followed by a listing of all the local variables in each frame.
    일반적인 역추적 정보를 인쇄한 다음 각 프레임의 모든 지역 변수 목록을 인쇄합니다.
    """
    try:
        assert False # intentional exception
    except AssertionError:
        tb = sys.exc_info()[2]

    while True:
        if not tb.tb_next:
            break
        tb = tb.tb_next

    stack = []
    f = tb.tb_frame.f_back
    log = []
    local_history = []

    while f:
        line_range = range(-line_window, line_window + 1)

        code_lines = []
        for line_no in line_range:
            # if line_no + f.f_lineno >= 1:
            l = linecache.getline(f.f_code.co_filename, f.f_lineno + line_no)
            if l != '':
                code_lines.append('%d | %s'%(f.f_lineno + line_no, l))

        code_lines = ''.join(code_lines)

        log.append('{}, {}:{}{}{}{}{}'.format(f.f_code.co_name, f.f_code.co_filename, f.f_lineno, os.linesep, os.linesep, code_lines, os.linesep))
        stack.append(f)
        local_history.append(f.f_locals)

        f = f.f_back

    log.reverse()
    local_history.reverse()
    frame = stack[0]

    # helper functions for debugging

    def extract_history(var_name):
        """Show the history of specific variable name.
        """

        res = {}

        for idx, hist in enumerate(local_history):
            if var_name in hist:
                print('=============')
                print(log[idx].split(os.linesep)[0])
                print('%s : %s (type %s)'%(var_name, hist[var_name], type(hist[var_name])))
                res[idx] = hist[var_name]

        tmp = []

        for k, v in res.items():
            for e in tmp:
                if type(e) == type(v) and e == v:
                    break
            else:
                tmp.append(v)

        if tmp == 1:
            return tmp[0]
        else:
            return res

    tools = {'extract_history' : extract_history,
             'history' : extract_history, }

    print(('================' + os.linesep).join(log))

    debugger_locals = {\
        'log' : log,
        'traceback_log' : ('================' + os.linesep).join(log),
        'debug_pos' : '{}, {}:{}'.format(frame.f_code.co_name, frame.f_code.co_filename, frame.f_lineno),
        'local_history' : local_history,
        'pprint' : pprint,
        'pformat' : pformat,
        **frame.f_globals,
        **frame.f_locals,
        **tools,
        **globals()}

    def run(file_name = 'tmp_test.py', tmp_locals = debugger_locals):
        cur_dir = os.getcwd()
        new_dir = os.sep.join(frame.f_code.co_filename.split(os.sep)[:-1])
        os.chdir(new_dir)
        code_file = open(file_name, 'r', encoding = 'utf-8').read()

        i = InteractiveInterpreter(locals = tmp_locals)
        i.runcode(code_file)
        os.chdir(cur_dir)

    interact(local = {'run' : run, **debugger_locals, })
