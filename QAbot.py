from langchain_community.llms import CTransformers
from langchain_google_genai import GoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import os

# Setup
load_dotenv()
vector_db_path = 'vectorstore'
# api_key = os.getenv("OPENAI_API_KEY")

# Load LLM
def load_llm():
    llm = GoogleGenerativeAI(model="models/gemini-1.5-pro-latest", google_api_key=os.environ["GOOGLE_API_KEY"], temperature=0.1)
        
    return llm

# Tạo prompt template để truyền prompt vào model
def create_prompt(template):
    prompt = PromptTemplate(template = template, input_variables=["context", "question"]) # context: thông tin ngữ cảnh của câu prompt
    
    return prompt

# Tạo QA chain: quy trình prompt -> llm model -> tìm kiếm similar trong db -> response
def create_QA_chain(prompt, llm, db):
    llm_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type = "stuff",
            input_key="query",
            retriever = db.as_retriever(score_threshold=0.7,
                                        max_tokens_limit=1024,
                                        # search_kwargs = {"k":3} # k: số đoạn văn bản gần nhất
                                        ),
            return_source_documents = True,
            chain_type_kwargs = {'prompt': prompt}
        )
    
    return llm_chain

# Read VectorDB
def read_vectors_db_from_dic():
    # Embeding
    embedding_model = GPT4AllEmbeddings(model_file="models/all-MiniLM-L12-v2.F16.gguf")
    # Đọc db từ file ở local
    db = FAISS.load_local(vector_db_path, embedding_model, allow_dangerous_deserialization=True)
    return db

# Testing
# db = read_vectors_db() # chỉ cần chạy 1 lần
# llm = load_llm()

# ## Tạo prompt
# template = """Với ngữ cảnh và câu hỏi được cung cấp, hãy chỉ tạo câu trả lời dựa trên ngữ cảnh này.
# Trong câu trả lời, hãy cố gắng cung cấp càng nhiều văn bản càng tốt từ phần 'phản hồi' trong tài liệu nguồn gốc mà không cần thay đổi nhiều.
# Nếu câu trả lời không được tìm thấy trong ngữ cảnh, vui lòng nói "Tôi không biết". Đừng cố gắng bịa ra câu trả lời.
#     CONTEXT: {context}
    
#     QUESTION: {question}"""
# prompt = create_prompt(template)

# llm_chain = create_QA_chain(prompt, llm, db)

# # Run
# question = "trung sinh năm bao nhiêu ?"
# response = llm_chain.invoke({"query": question})
# print(response)

