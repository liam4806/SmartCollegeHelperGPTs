from langchain.chat_models import ChatOpenAI, ChatOllama
from langchain.document_loaders import UnstructuredFileLoader
from langchain.schema.output import ChatGenerationChunk, GenerationChunk
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings, OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.storage import LocalFileStore
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
import streamlit as st
from langchain.callbacks.base import BaseCallbackHandler
from langchain.memory import ConversationSummaryBufferMemory
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain.chains import LLMChain
from operator import itemgetter
from langchain_community.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
import pages.DocumentGPT as docuGPT
from pages.DocumentGPT import send_message, paint_history, save_memory, save_message

st.set_page_config(
    page_title = "Therapist GPT",
    page_icon = "T"
)

st.title("Therapist GPT")

def send_message(message, role, save = True):
    with st.chat_message(role):
        st.markdown(message)
    if(save == True):
        save_message(message, role)
        
def paint_history():
    for message in st.session_state["messages"]:
        send_message(message["message"],message["role"], save = False)
    
def save_message(message,role):
    st.session_state["messages"].append({"message" : message, "role" : role})

def save_memory(input, output):
    st.session_state["memory"].save_context({"input" : input},{"output" : output})


st.markdown("""
Welcome!

I am your Therapist. Let me know any concerns you have.\n

You can choose three different models.\n
There are two Private Therapist Fine-tuned models with different numbers of datasets trained. The conversation will run completely locally and privately. \n
Another option is using Chat GPT3.5 which is faster but the conversations will be sent to the OpenAI server.

""")

callback_manager = CallbackManager([docuGPT.ChatCallbackHandler()])

with st.sidebar:
    choice = st.selectbox(
        "Choose the Chat Model you want.",
        (
            "Private Therapist Fine-Tuned(3.5k)",
            "Private Therapist Fine-Tuned(824k)",
            "GPT3.5 (Public, Faster)",
        ),
    )
    if(choice == "Private Therapist Fine-Tuned(3.5k)"):
        n_gpu_layers = -1
        n_batch = 512
        llm = LlamaCpp(
            model_path = "/Users/liamyoun/Desktop/FullStack_GPT/models/llama3_therapist_3k_tuned.gguf",
            n_gpu_layers = n_gpu_layers,
            n_batch = n_batch,
            f16_kv = True,  
            callback_manager = callback_manager,
            verbose = True,  
            n_ctx = 8000,
        )
        memory_llm = ChatOllama(
            model = "llama3:latest",
            temperature = 0.1
        )
        
        template = ChatPromptTemplate.from_template("""
        <|begin_of_text|>
        <|start_header_id|>system<|end_header_id|>
        You are a helpful AI psychotherapist. 
        Anaylze carefully about the user's situation and provide your best analysis and suggestions to user as much as possible. Provide long answers.
        Below is the chat history.
        Use the contents below you answer to check only if there are any relative information that the user mentioned or the user ask about the previous conversation.
        chat history: {chat_history}
        <|eot_id|>
        <|start_header_id|>user<|end_header_id|>
        {question}
        <|eot_id|>
        <|start_header_id|>assistant<|end_header_id|>
            """)
    elif(choice == "Private Therapist Fine-Tuned(824k)"):
        n_gpu_layers = -1
        n_batch = 512
        llm = LlamaCpp(
            model_path = "/Users/liamyoun/Desktop/FullStack_GPT/models/llama3_therapist_824k_tuned.gguf",
            n_gpu_layers = n_gpu_layers,
            n_batch = n_batch,
            f16_kv = True,  
            callback_manager = callback_manager,
            verbose = True,  
            n_ctx = 8000,
        )
        
        memory_llm = ChatOllama(
            model = "llama3:latest",
            temperature = 0.1
        )
        
        template = ChatPromptTemplate.from_template("""
        <|begin_of_text|>
        <|start_header_id|>system<|end_header_id|>
        You are a helpful AI psychotherapist. 
        Anaylze carefully about the user's situation and provide your best analysis and suggestions to user as much as possible. Provide long answers.
        Below is the chat history.
        Use the contents below you answer to check only if there are any relative information that the user mentioned or the user ask about the previous conversation.
        chat history: {chat_history}
        <|eot_id|>
        <|start_header_id|>user<|end_header_id|>
        {question}
        <|eot_id|>
        <|start_header_id|>assistant<|end_header_id|>
            """)
    else:
        llm = ChatOpenAI(
        temperature = 0.3, streaming = True,
        callbacks = [
            docuGPT.ChatCallbackHandler(),
        ])
        memory_llm = ChatOpenAI(temperature= 0.1)
        
        template = ChatPromptTemplate.from_messages([
            ("system", """
            You are a helpful AI psychotherapist. Be polite to the other person as possible.
            Give your thoughts as much as possible. Respond with the same language of the user.
            Below is the chat history for you to talk like a conversational AI.
            Use the contents below only if you need to talk about it or the user asked.
            """),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}")
        ])      
print(llm)
if "messages" not in st.session_state:
    st.session_state["messages"] = []
    
if "memory" not in st.session_state:
    st.session_state["memory"] = ConversationSummaryBufferMemory(
        llm = memory_llm, max_token_limit = 400, return_messages = True, memory_key = "chat_history", input_key = "question"
    )
send_message("Tell me anything!", "ai", save = False)
paint_history()
message = st.chat_input("Chat")
if message:
    send_message(message, "human")
    chain = LLMChain(
        llm = llm,
        memory = st.session_state["memory"],
        prompt= template,
        verbose = True
    )
    with st.chat_message("ai"):
        response = chain.invoke({"question" : message, "chat_history": st.session_state.memory.load_memory_variables({})})



