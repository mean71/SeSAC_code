from pprint import pprint 
from debugger import debug_shell
from openai import OpenAI
from logging_util import log_chatgpt_call

openai_api_key = 

client = OpenAI(api_key = openai_api_key)

@log_chatgpt_call
def ask_chatgpt(
    query: str, 
    model:str = 'gpt-4o-mini-2024-07-18',
):
    completion = client.chat.completions.create(
        model=f"{model}",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": f"{query}"
            }
        ]
    )

    return completion 

if __name__ == '__main__':
    completion = ask_chatgpt('Design a repository structure for making interactive neural net builder, that is capable of 1) building neural net structure by drag/drop, 2) compile the structure into pytorch code and export it, and 3) control the hyperparameters for the neural net automatically.')
    pprint(completion)
    # print(completion.choices[0].message)