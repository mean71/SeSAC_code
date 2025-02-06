import os

config_file_dir = os.sep.join(os.path.abspath(__file__).split(os.sep)[:-1])

DATA_DIR = os.path.join(config_file_dir, 'data')
en2fr_data = os.path.join(DATA_DIR, 'eng-fra.txt')

function_execution_log = 'function_execution_log.txt'


'''
os.path.abspath(__file__)
-> 해당코드가 실행되는 파일의 절대 경로 반환

.split(os.sep)
-> 운영체제에 맞는 경로 구분자  window에서는 '\\' unix에서는 '/ '

config_file_dir = os.sep.join(os.path.abspath(__file__).split(os.sep)[:-1])
-> 절대 경로를 구분자로 나누어 리스트로 만들고 리스트의 마지막 요소(현재 파일명)을 제외한 나머지 부분을 가져옴

os.sep.join( os.path.abspath(__file__).split(os.sep)[:-1] )
-> 리스트의 요소들에서 파일명을 제외하고 다시 경로구분자로 연결하여 운영체제와 무관하게 현재 파일의 디렉토리 경로를 추출하여 해당디렉토리내 파일과 폴더에 접근하는데 사용
'''