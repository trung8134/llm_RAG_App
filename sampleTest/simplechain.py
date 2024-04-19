from langchain_community.llms import CTransformers
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Setup
model_file = 'models/vinallama-7b-chat_q5_0.gguf'

# Load LLM
def load_file(model_file):
    llm = CTransformers(
        model = model_file,
        model_type = 'llama',
        max_new_tokens = 1024, # số lượng token mới sinh ra
        temperature = 0.01 # độ sáng tạo của model
    )
    
    return llm

# Tạo prompt template để truyền prompt vào model
def create_prompt(template):
    prompt = PromptTemplate(template = template, input_variables=["question"])
    
    return prompt

# Tạo simple chain: quy trình prompt -> llm model -> response
def create_simple_chain(prompt, llm):
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    
    return llm_chain

# Chạy thử chain

template = """<|im_start|>system
Bạn là một trợ lí AI hữu ích. Hãy trả lời người dùng một cách chính xác.
<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant"""

prompt = create_prompt(template)
llm = load_file(model_file)
llm_chain = create_simple_chain(prompt, llm)

# Testing
question = "Bác Hồ sinh năm bao nhiêu"
response = llm_chain.invoke({"question":question}) 
print(response)