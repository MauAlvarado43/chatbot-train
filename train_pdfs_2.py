# Train a GPT-3 model on a directory of PDFs
# Low-Mid level API (embeddings control)

from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter # 0.0.197
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import DeepLake
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
import PyPDF2
import glob
import os

os.environ["OPENAI_API_KEY"] = ''

pdfs = glob.glob('./docs/*.pdf')
dataset_path = './dataset'
extracted = []

# Extract text from pdfs
for pdf in pdfs:
    pdf_file = open(pdf, 'rb')
    pdf_reader = PyPDF2.PdfReader(pdf_file)

    page_text = ''
    
    for i in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[i]
        page_text += page.extract_text()

    pdf_file.close()

    extracted.append({
        'file': pdf,
        'text': page_text
    })

# Create embeddings from our pdfs
embeddings = OpenAIEmbeddings()
db = DeepLake(dataset_path="./dataset/", embedding_function=embeddings)

for pdf in extracted:

    # Split and make pages from our pdf
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    pages = text_splitter.split_text(pdf['text'])

    # Create texts from pages
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.create_documents(pages)

    # Create embeddings at our database
    db.add_documents(texts, metadata={'file': pdf['file']})

# read from our database
db = DeepLake(dataset_path=dataset_path, read_only=True, embedding_function=embeddings)

# retrieve the data
retriever = db.as_retriever()
retriever.search_kwargs['distance_metric'] = 'cos'
retriever.search_kwargs['k'] = 4

qa = RetrievalQA.from_chain_type(
    llm=OpenAI(), 
    chain_type="stuff", 
    retriever=retriever, 
    return_source_documents=False
)

# ask a question
while True:
    query = input("")
    ans = qa({"query": query})
    print(ans)