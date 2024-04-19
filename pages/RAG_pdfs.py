import streamlit as st
from dotenv import load_dotenv
import os
import shutil
from prepare_vector_DB import create_db_from_pdf_local_files, create_db_from_pdf_files
from QAbot import read_vectors_db_from_dic, create_QA_chain, load_llm, create_prompt

# Setup
DIRECTORY_PDF = "data"
DIRECTORY_VECTOR_DB = "vectorstore"
template = """Sử dụng thông tin sau đây để trả lời câu hỏi. 
Nếu không tìm thấy câu trả lời trong ngữ cảnh, vui lòng nói "Tôi không biết". Đừng cố bịa ra một câu trả lời.
    CONTEXT: {context}
    
    QUESTION: {question}"""
    

def delete_files(dir):
    for filename in os.listdir(dir):
        file_path = os.path.join(dir, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('')
            
            
def handle_userinput(user_question):
    response = st.session_state.conversation
    response_n = response.invoke({"query": user_question})
    st.write(response_n['result'])
            
def main():
    # setup apikey from .env
    load_dotenv()
    # set title 
    st.set_page_config(page_title='Chat with Multiple PDFs', page_icon='🤖', layout='wide', initial_sidebar_state='auto')
    # setup input 
    st.header("Chat with Multiple PDFs")
    user_question = st.text_input("Ask me anything about the PDFs", key='question')
    
    # output
    if user_question:
        handle_userinput(user_question)
    else:
        st.text("Please enter your question to run")
    
    # sidebar
    with st.sidebar:
        st.subheader("Your doccuments")
        ## xóa các file pdf, vector db cũ trong thư mục khi reload web (nếu có lưu dữ liệu trong thư mục)
        # delete_files(DIRECTORY_PDF)
        # delete_files(DIRECTORY_VECTOR_DB)
        
        ## upload file pdf
        pdf_docs = st.file_uploader(
            "Upload your PDFs and click Process button", 
            type=['pdf'], 
            accept_multiple_files=True, 
            key='pdfs')
        
        ## Lưu dữ liệu tải lên vào thư mục data (dùng khi cần lưu trữ file pdf vào 1 thư mực)
        # if pdf_docs is not None:
        #     for pdf_doc in pdf_docs:
        #         with open(os.path.join(DIRECTORY_PDF, pdf_doc.name), "wb") as f:
        #             f.write(pdf_doc.getbuffer())
                    
        if st.button("Process PDFs", key='process_pdfs'):
            with st.spinner("Processing PDFs..."):
                # vector db
                # db = create_db_from_files() # tạo vector db từ các file pdf trong thư mục data và lưu vào thư mục vectorstore
                # db = read_vectors_db() # đọc vector db từ thư mục vectorstore
                db = create_db_from_pdf_files(pdf_docs) # tạo vector db từ file pdf 
                
                # create llm model
                llm = load_llm()
                
                # create prompt template
                prompt = create_prompt(template=template)
                
                # create conversation chain
                st.session_state.conversation = create_QA_chain(prompt, llm, db) # lưu trữ llm_chain

                # Add a message to indicate that llm_chain is ready
                st.success("llm_chain is ready!")
                
if __name__ == '__main__':
    main()



