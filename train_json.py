# Train a GPT-3 model with a JSONL file

import openai
import requests
import json

# Set OpenAI API key

openai.api_key = ''

################
# Prepare data #
################

# Be careful with the format of the data. It must be a list of dictionaries with the following format:
# [
#     {
#         "prompt": "Example prompt->",
#         "completion": " Example completion\n"
#     }
# ]

# The completion must end with a new line character (\n) to avoid concatenating the next prompt with the previous completion.
# The prompt must end with a -> to avoid concatenating the previous prompt with the next completion.

training_data = [
    {
        "prompt": "Example prompt->",
        "completion": " Example completion\n"
    }
]

validation_data = [
    {
        "prompt": "Example prompt->",
        "completion": " Example completion\n"
    }
]

training_file_name = "training_data.jsonl"
validation_file_name = "validation_data.jsonl"

def prepare_data(dictionary_data, final_file_name):
    with open(final_file_name, 'w') as outfile:
        for entry in dictionary_data:
            json.dump(entry, outfile)
            outfile.write('\n')

prepare_data(training_data, "training_data.jsonl")
prepare_data(validation_data, "validation_data.jsonl")

# Execute in terminal:
# openai tools fine_tunes.prepare_data -f "training_data.jsonl"
# openai tools fine_tunes.prepare_data -f "validation_data.jsonl"

################
# Upload files #
################

def upload_data_to_OpenAI(file_name):
    file = openai.File.create(file=open(file_name), purpose='fine-tune')
    return file.id

training_file_id = upload_data_to_OpenAI(training_file_name)
validation_file_id = upload_data_to_OpenAI(validation_file_name)

print(f"Training file id: {training_file_id}")
print(f"Validation file id: {validation_file_id}")

################
# Fine-tunning #
################

create_args = {
	"training_file": "", # training_file_id
	"validation_file": "", # validation_file_id
	"model": "davinci",
	"n_epochs": 15,
	"batch_size": 3,
	"learning_rate_multiplier": 0.3
}

response = openai.FineTune.create(**create_args)
job_id = response["id"]
status = response["status"]

print(f'Fine-tunning model with jobID: {job_id}.')
print(f"Training Response: {response}")
print(f"Training Status: {status}")

################
#   FT status  #
################

ft_id = "" # job_id

import time

status = openai.FineTune.retrieve(id=ft_id)["status"]

if status not in ["succeeded", "failed"]:
    
    print(f'Job not in terminal status: {status}. Waiting.')

    while status not in ["succeeded", "failed"]:
        time.sleep(2)
        status = openai.FineTune.retrieve(id=ft_id)["status"]
        print(f'Status: {status}')

else:

    print(f'Finetune job {ft_id} finished with status: {status}')

print('Checking other finetune jobs in the subscription.')

result = openai.FineTune.list()

print(f'Found {len(result.data)} finetune jobs.')
print(result)

ft_model = "" # result.data[0].model

################
#   FT model   #
################

new_prompt = "" # "Example prompt
answer = openai.Completion.create(
  model=ft_model,
  prompt=new_prompt
)

print(answer['choices'])