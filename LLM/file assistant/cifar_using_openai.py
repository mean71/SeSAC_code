import base64
import json 
import os
import pickle
import torch
from io import BytesIO
from openai import OpenAI 
from PIL import Image
from torchvision import datasets, transforms

from debugger import debug_shell 
from logging_util import log_chatgpt_call 

def make_base64_cifar10():
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
    cifar10_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)

    cifar10_base64 = []

    for img_tensor, label in cifar10_dataset:
        img_base64 = image_to_base64(img_tensor)
        cifar10_base64.append({
            "label": cifar10_dataset.classes[int(label)],
            "image_base64": img_base64
        })

    return cifar10_base64

# Function to convert an image tensor to base64
def image_to_base64(image_tensor):
    # Convert tensor to PIL Image
    image = transforms.ToPILImage()(image_tensor).convert("RGB")
    
    # Save the image to a BytesIO object
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    
    # Encode as base64
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str


@log_chatgpt_call
def classify_image(base64_image):

    openai_api_key = ''
    
    client = OpenAI(api_key = openai_api_key)
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                    "type": "text",
                    "text": "You are now going to perform image classification task. You have to choose the right label for the attached image. You have to choose between 10 options; airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.",
                    },
                    {
                    "type": "image_url",
                    "image_url": {
                        "url":  f"data:image/jpeg;base64,{base64_image}"
                    },
                    },
                ],
            }  
        ],
        response_format = {
            'type' : 'json_schema', 
            'json_schema': {
                'name': 'cifar_result', 
                'schema': {
                    'type': 'object', 
                    'properties': {
                        'classification_result': {
                            'description': 'Single word about cifar10 image classification result chosen among ten options; airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.', 
                            'type': 'string', 
                        }, 
                    }
                }
            }
        }
    )

    return response

if __name__ == '__main__':
    cifar_pickle = 'cifar10.base64.pickle' 

    if cifar_pickle in os.listdir():
        cifar_images = pickle.load(open(cifar_pickle, 'rb'))
    else:
        cifar_images = make_base64_cifar10()
        pickle.dump(cifar_images, open(cifar_pickle, 'wb+'))
    
    print(len(cifar_images))
    correct = 0 
    total = 100
    for elem in cifar_images[:total]:
        label = elem['label']
        image = elem['image_base64']
        response = classify_image(image)

        print(response.choices[0].message.content, label)
        d = json.loads(response.choices[0].message.content)
        if 'classification_result' in d:
            if d['classification_result'] == label:
                correct += 1 
    print('Accuracy: ', correct / total)
    debug_shell()


