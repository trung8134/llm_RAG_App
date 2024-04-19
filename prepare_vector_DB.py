# import lib 2 trường hợp tách câu
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
# import lib load nội dung file pdf và load toàn bộ file trong 1 folder
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
# import lib vector DB
from langchain_community.vectorstores import FAISS
# import lib embedding text
from langchain_community.embeddings.gpt4all import GPT4AllEmbeddings
from langchain_community.embeddings.huggingface import HuggingFaceInstructEmbeddings
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
import requests
from bs4 import BeautifulSoup

# Khai báo biến
data_path = 'data'
vector_db_path = 'vectorstore'
load_dotenv()

### Tạo vector DB từ 1 thư mục chứa dữ liệu pdf
def create_db_from_pdf_local_files():
    ## Khai báo loader để quét toàn bộ thư mục data
    loader = DirectoryLoader(data_path, glob="*.pdf", loader_cls = PyPDFLoader)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)

    ## Embeding
    embedding_model = GPT4AllEmbeddings(model_file="models/all-MiniLM-L12-v2.F16.gguf")
    # embedding_model = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")

    db = FAISS.from_documents(chunks, embedding_model)
    db.save_local(vector_db_path) # lưu db vào thư mục vectorstore
    return db


### Tạo vector DB từ 1 file pdf
def create_db_from_pdf_files(pdf_docs):
    ## Lấy text từ file pdf
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    
    ## Chunk text thành các đoạn nhỏ
    text_splitter = CharacterTextSplitter(
        separator='\n',
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = text_splitter.split_text(text)

    ## Embeding
    embedding_model = GPT4AllEmbeddings(model_file="models/all-MiniLM-L12-v2.F16.gguf")
    # embedding_model = HuggingFaceInstructEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2")

    db = FAISS.from_texts(chunks, embedding_model)
    return db

### Tạo vector DB từ 1 url website
def create_db_from_website_url(url):
    ## Lấy text từ 1 website
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    text = soup.get_text()
    
    ## Chunk text thành các đoạn nhỏ
    text_splitter = CharacterTextSplitter(
        separator='\n',
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = text_splitter.split_text(text)

    ## Embeding
    embedding_model = GPT4AllEmbeddings(model_file="models/all-MiniLM-L12-v2.F16.gguf")
    # embedding_model = HuggingFaceInstructEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2")

    db = FAISS.from_texts(chunks, embedding_model)
    return db