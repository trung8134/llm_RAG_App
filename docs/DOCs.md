All we need ?
- LLM model: vinallama-7b-chat_q5_0
- PDF files
- Langchain: framework working on LLM 
- Faiss Vector DB
- CTransformer: Run Quantization on CPU
- .env file: setup API key and ....

Bug when run code:
- Using gpt4all 2.2.1

st.session_state.conversation: dùng để lưu trữ 1 biến hay 1 hàm, class cần lưu trữ trong cache. Lấy ra sử dụng bất cứ lúc nào.