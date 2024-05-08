# Upload fine-tuning files

import os
from openai import AzureOpenAI, FineTuningJob
from setupenv import setup

setup()

client = AzureOpenAI(
  azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"), 
  api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
  api_version="2024-02-01"  # This API version or later is required to access fine-tuning for turbo/babbage-002/davinci-002
)

def upload_file(filename: str) -> str:
    training_response = client.files.create(file=open(filename, "rb"), purpose="fine-tune")
    return training_response.id

def fine_tune(training_file_id: str) -> FineTuningJob:
    response = client.fine_tuning.jobs.create(
        training_file=training_file_id,
        model="gpt-35-turbo-0613", hyperparameters={"n_epochs": 2})
    return response

def check_job_status(job_id: str) -> FineTuningJob:
    response = client.fine_tuning.jobs.retrieve(job_id)
    return response

def print_status(response: FineTuningJob):
    print("Job ID:", response.id)
    print("Status:", response.status)
    print(response.model_dump_json(indent=2))


####
# training_file_id = upload_file('training_set.jsonl')
training_file_id = "file-6550c822e0864a4e8c6da08675201e9c"

print("Training file ID:", training_file_id)

# job = fine_tune(training_file_id)
job = check_job_status("ftjob-11098c6a489f4cf5b490a4868a395e66")
print_status(job)

