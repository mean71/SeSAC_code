# -*- coding: utf-8 -*

from pprint import pprint 
from debugger import debug_shell
from openai import OpenAI
from logging_util import log_chatgpt_call

openai_api_key = 
max_communicaton_length = 3

client = OpenAI(api_key = openai_api_key)

@log_chatgpt_call
def make_pronounciation(
    model:str = 'gpt-4o-mini-2024-07-18',
):

    words = ['hello', 'world', 'dollar', 'won']
    korean_words = ['헬로우 (hello)', '월드 (world)', '달러 (dollar)', '원 (won)'] 
    new_words = ['how', 'do', 'you', 'do']

    completion = client.chat.completions.create(
        model = f'{model}',
        messages = [
            {
                'role': 'system', 
                'content': 'You are a helpful assistant.', 
            },
            {
                'role': 'user',
                'content': f'You will be provided with a list of english words. I want you to make a Korean pronounciation of each words.\n\n' + '\n- '.join(words),
            }, 
            {
                'role': 'assistant', 
                'content' : '- ' + '\n- '.join(korean_words), 
            }, 
            {
                'role': 'user', 
                'content': '- ' + '\n- '.join(new_words), 
            },
        ]
    )

    return completion 

@log_chatgpt_call
def step_conversation(
    context, 
    model:str = 'gpt-4o-mini-2024-07-18',
):  
    if len(context) > max_communicaton_length:
        return 
    completion = client.chat.completions.create(
        model = f'{model}', 
        messages = context, 
    )

    return completion

def debate_simulation(
    agenda = '닭이 먼저냐 달걀이 먼저냐, 닭이 먼저이다.', 
    pro_model = 'gpt-4o-mini-2024-07-18', 
    con_model = 'gpt-3.5-turbo-0125', 
):

    initial_context_pro = [
        {
            'role': 'system', 
            'content': f'너는 매우 한국어에 능통한 철학 및 생물학 전문가야. 이 때, 주어진 아젠다인 "{agenda}"에 대해서, 닭이 먼저라고 강하게 생각하고 있고 이것을 달걀이 먼저라고 생각하는 사람에게 굉장히 잘 설득할 수 있는 능력이 있어. 동시에, 너는 굉장히 포용적이라 상대방이 말한 말이 맞다면 너의 원래 생각을 바꾸고 달걀이 먼저라고 인정할 수 있어. 만약 너의 생각이 바뀌었다면 "패배를 인정합니다" 라고 말하고 더 답변하지 않을 거야.'  
        }, 
        {
            'role': 'user', 
            'content': f'이제 "{agenda}"에 대한 토론을 시작해봐.', 
        }, 
    ]

    initial_context_con = [
        {
            'role': 'system', 
            'content': f'너는 매우 한국어에 능통한 철학 및 생물학 전문가야. 이 때, 주어진 아젠다인 "{agenda}"에 대해서, 달걀이 먼저라고 강하게 생각하고 있고 이것을 닭이 먼저라고 생각하는 사람에게 굉장히 잘 설득할 수 있는 능력이 있어. 동시에, 너는 굉장히 포용적이라 상대방이 말한 말이 맞다면 너의 원래 생각을 바꾸고 닭이 먼저라고 인정할 수 있어. 만약 너의 생각이 바뀌었다면 "패배를 인정합니다" 라고 말하고 더 답변하지 않을 거야.' 
        }, 
        {
            'role': 'user', 
            'content': f'이제 "{agenda}"에 대한 토론을 시작해봐.', 
        }, 
    ]

    def generate_messages(context, is_pro):
        for idx, message in enumerate(context):
            if message['role'] == 'system' and idx == 0:
                if is_pro: 
                    context[idx]['content'] = initial_context_pro[0]['content']
                else: 
                    context[idx]['content'] = initial_context_con[0]['content']
            
            if idx % 2 == 1: 
                if is_pro:
                    context[idx]['role'] = 'assistant'
                else:
                    context[idx]['role'] = 'user'
            else:
                if is_pro:
                    context[idx]['role'] = 'user'
                else:
                    context[idx]['role'] = 'assistant'
        return context 
  
    context = initial_context_pro
    pro = True

    while len(context) < max_communicaton_length:
        context = generate_messages(context, pro)
        try:
            completion = step_conversation(
                context, 
                model = pro_model if pro else con_model, 
            )
        except:
            debug_shell()
        context.append({
            'role' : 'assistant', 
            'content' : str(completion.choices[0].message.content)
        })

        print('=======================')
        print(str(completion.choices[0].message.content))

        pro = not pro 

if __name__ == '__main__':
    translate('안녕하세요')
    # >> 'Hello'
    # print(completion.choices[0].message)