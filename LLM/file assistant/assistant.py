# file search using openai assistant
from time import time 

from openai import OpenAI 

def ask(prompt):
    openai_api_key = ''
    client = OpenAI(api_key = openai_api_key)
    
    begin = time()

    assistant = client.beta.assistants.create(
        name="git professional",
        instructions="You are an expert in git and overall IT infrastructure.",
        model="gpt-4o-mini-2024-07-18",
        tools=[{"type": "file_search"}],
    )

    file_obj = client.files.create(
                file = open('progit (1).pdf', 'rb'),
                purpose = 'assistants'
            )

    thread = client.beta.threads.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
                # Attach the new file to the message.
                "attachments": [
                    { 
                        "file_id": file_obj.id, 
                        "tools": [{"type": "file_search"}], 
                    }
                ],
            }
        ]
    )

    run = client.beta.threads.runs.create_and_poll(
        thread_id=thread.id, assistant_id=assistant.id
    )

    messages = list(client.beta.threads.messages.list(thread_id=thread.id, run_id=run.id))

    message_content = messages[0].content[0].text
    annotations = message_content.annotations
    citations = []
    
    for index, annotation in enumerate(annotations):
        message_content.value = message_content.value.replace(annotation.text, f"[{index}]")
        if file_citation := getattr(annotation, "file_citation", None):
            cited_file = client.files.retrieve(file_citation.file_id)
            citations.append(f"[{index}] {cited_file.filename}")

    print(message_content.value)
    print("\n".join(citations))

if __name__ == '__main__':
    ask('git이 무엇이고 이것을 왜 써야 하는지 파일을 참고해서 자세하게 설명해봐.')
