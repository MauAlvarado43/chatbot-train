# Train a GPT-3 model on a directory of PDFs
# Medium level API (embeddings control + transformer model)

import pandas as pd
import PyPDF2
import spacy
import glob
import numpy as np
import openai
import tiktoken
import time
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize

# Load models

spacy.prefer_gpu()
nlp = spacy.load('es_core_news_lg')

sentence_tf_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

openai.api_key = ''

COMPLETIONS_MODEL = "text-davinci-003"
EMBEDDING_MODEL = "text-embedding-ada-002"
GPT_MODEL = "gpt-3.5-turbo"

MAX_SECTION_LEN = 500
SEPARATOR = "\n* "
ENCODING = "gpt2"  # encoding for text-davinci-003

COMPLETIONS_API_PARAMS = {
    # We use temperature of 0.0 because it gives the most predictable, factual answer.
    "temperature": 0.0,
    "max_tokens": 1000,
    "model": COMPLETIONS_MODEL,
}

# Prepare data

encoding = tiktoken.get_encoding(ENCODING)
separator_len = len(encoding.encode(SEPARATOR))

pdfs = glob.glob('./docs/*.pdf')
df = pd.DataFrame(columns=['pdf', 'page', 'content'])

# Extract text from pdfs
for pdf in pdfs:

    pdf_reader = PyPDF2.PdfReader(pdf)
    num_pages = len(pdf_reader.pages)
    paragraphs = []
    
    for i in range(num_pages):

        page = pdf_reader.pages[i]
        text = page.extract_text()

        doc = nlp(text)
        sentences = [sent.text for sent in doc.sents]

        paragraph = ''

        for sentence in sentences:
            if len(paragraph) + len(sentence) > 500:
                paragraphs.append(paragraph)
                paragraph = sentence
            else:
                paragraph += ' ' + sentence

        paragraphs.append(paragraph)

    for i in range(len(paragraphs)):
        df.loc[len(df.index)] = [pdf, i, paragraphs[i]]

# Compute embeddings from GPT-3
def get_embedding(text, model = EMBEDDING_MODEL):
    result = openai.Embedding.create(model = model, input = text)
    time.sleep(5)
    return result["data"][0]["embedding"]

# Get embeddings from GPT-3 for every document
def compute_doc_embeddings(df):
    return { idx: get_embedding(r.content) for idx, r in df.iterrows() }

# Load embeddings from csv
def load_embeddings(fname):
    df = pd.read_csv(fname, header=0)
    max_dim = max([int(c) for c in df.columns if c != "title" and c != "heading"])
    return { (r.title, r.heading): [r[str(i)] for i in range(max_dim + 1)] for _, r in df.iterrows() }

# Compute embeddings from Transformers model
def get_hf_embeddings(text, model):
    sentence_embeddings = model.encode(text)
    sentence_embeddings = sentence_embeddings.reshape(1, -1)
    sentence_embeddings = normalize(sentence_embeddings)
    return sentence_embeddings[0]

# Get embeddings from Transformers model for every document
def compute_doc_embeddings_hf(df, model):
    return { idx: get_hf_embeddings(r.content, model) for idx, r in df.iterrows() }

# Compute similarity between two vectors
def vector_similarity(x, y):
    return np.dot(np.array(x), np.array(y))

# Order document sections by similarity to query
def order_document_sections_by_query_similarity(query, contexts):
    
    query_embedding = get_hf_embeddings(query, sentence_tf_model)
    # query_embedding = get_embedding(query)

    document_similarities = sorted([
        (vector_similarity(query_embedding, doc_embedding), doc_index) for doc_index, doc_embedding in contexts.items()
    ], reverse=True)

    return document_similarities

# Construct prompt for model
def construct_prompt(question, context_embeddings, df):

    most_relevant_document_sections = order_document_sections_by_query_similarity(question, context_embeddings)

    chosen_sections = []
    chosen_sections_len = 0
    chosen_sections_indexes = []

    for _, section_index in most_relevant_document_sections:
        document_section = df.loc[section_index]
        document_tokens = len(encoding.encode(document_section.content))
        chosen_sections_len += document_tokens + separator_len
        if chosen_sections_len > MAX_SECTION_LEN:
            break

        chosen_sections.append(SEPARATOR + document_section.content.replace("\n", " "))
        chosen_sections_indexes.append(str(section_index))

    print(chosen_sections)
    print("\n".join(chosen_sections_indexes))

    return "".join(chosen_sections) + "\n\n Q: " + question + "\n A:"
    
def answer_query_with_context(query, df, document_embeddings, show_prompt):

    prompt = construct_prompt(
        query,
        document_embeddings,
        df
    )

    if show_prompt:
        print(prompt)

    response = openai.Completion.create(
        prompt=prompt,
        **COMPLETIONS_API_PARAMS
    )

    return response["choices"][0]["text"].strip(" \n")

# embeddings = compute_doc_embeddings(df)
embeddings_hf = compute_doc_embeddings_hf(df, sentence_tf_model)
prompt = construct_prompt(
    "",
    embeddings_hf,
    df
)

# answer_query_with_context("", df, embeddings_hf)

# manuel_res = openai.Completion.create(prompt="", **COMPLETIONS_API_PARAMS)
# manuel_res["choices"][0]["text"].strip(" \n")