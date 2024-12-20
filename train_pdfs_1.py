# Train a GPT-3 model on a directory of PDFs
# Low level API

from gpt_index import SimpleDirectoryReader, GPTListIndex, GPTSimpleVectorIndex, LLMPredictor, PromptHelper
from langchain.chat_models import ChatOpenAI # 0.0.118
import sys
import os

os.environ["OPENAI_API_KEY"] = ''

def construct_index(directory_path):

    max_input_size = 4096
    num_outputs = 512
    max_chunk_overlap = 20
    chunk_size_limit = 600

    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit = chunk_size_limit)

    llm_predictor = LLMPredictor(llm = ChatOpenAI(temperature = 0.7, model_name = "gpt-3.5-turbo", max_tokens = num_outputs))

    documents = SimpleDirectoryReader(directory_path).load_data()

    index = GPTSimpleVectorIndex(documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper)

    index.save_to_disk('index.json')

    return index

def chatbot(input_text):
    index = GPTSimpleVectorIndex.load_from_disk('index.json')
    response = index.query(input_text, response_mode = "compact")
    return response.response

index = construct_index("docs")
print(chatbot(""))