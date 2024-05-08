import os
from setupenv import setup
from openai import AzureOpenAI
from termcolor import colored

setup()
client = AzureOpenAI(
  azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"), 
  api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
  api_version="2024-02-01"
)

messages=[
    {"role": "system", "content": "You are a helpful pirate assistant."},
]

questions=[
    {"role": "user", "content": "Where is Milwaukee?"},
    {"role": "user", "content": "How far away is it from St Paul?"},
    {"role": "user", "content": "What is the best activity to do in St Paul?"},
    {"role": "user", "content": "What is the best drink in St Paul?"},
]

def print_user(message: str):
    print(colored(message, 'green'))

def print_ai(message: str):
    print('\t', colored(message, 'blue'))

def append_ai_response(messages: list[dict[str,str]], response):
    messages.append({"role": response.role, "content": response.content})
    return messages

def completion(messages: list[dict[str,str]], model: str):
    response = client.chat.completions.create(
        model=model, 
        messages=messages
    )

    return response.choices[0].message

def conversation(model: str):
    current_messages = messages.copy()

    for message in questions:
        current_messages.append(message)
        print_user(message['content'])
        response = completion(current_messages, model)
        current_messages = append_ai_response(current_messages, response)
        print_ai(response.content)

print("Regular GPT Response:")
conversation("gpt-35")

print("Pirate GPT Response:")
conversation("gpt-35-pirate")