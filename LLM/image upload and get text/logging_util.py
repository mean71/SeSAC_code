import os 
import random 
import time
from datetime import datetime

currency = 1400 

price_table = {
    'gpt-4o-mini-2024-07-18' : {
        'prompt_token' : 0.15/1000000, 
        'completion_token' : 0.6/1000000, 
    }, 
    'o1-preview-2024-09-12' : {
        'prompt_token' : 15/1000000, 
        'completion_token' : 60/1000000, 
    }, 
    'gpt-3.5-turbo-0125' : {
        'prompt_token' : 0.5/1000000, 
        'completion_token' : 1.5/1000000, 
    }, 
}

logging = 'log.txt' 

def render_result(completion):
    d = completion.to_dict()
    return d['choices'][0]['message']['content']

def calculate_token_usage(completion):
    d = completion.to_dict()

    model = d['model']
    prompt_token = d['usage']['prompt_tokens']
    completion_token = d['usage']['completion_tokens']

    ppt_prompt = price_table[model]['prompt_token']
    ppt_completion = price_table[model]['completion_token']

    return prompt_token * ppt_prompt + completion_token * ppt_completion 

def log_chatgpt_call(f):
    def func(*args, **kargs):
        begin = time.time()
        completion = f(*args, **kargs)
        time.sleep(random.random() + 0.5) 
        end = time.time() 

        expected_pricing = calculate_token_usage(completion) * currency
        call_results = render_result(completion) 

        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_line = os.linesep.join([
            f"{current_time} | Expected Pricing: {expected_pricing:.4f} | Time elapsed: f{round(end - begin, 3)}", 
            f"Results: {call_results}"
            ]) + os.linesep

        print(log_line, file = open(logging, 'a+'))

        return completion 
    return func 
    





    