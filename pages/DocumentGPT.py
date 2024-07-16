from uuid import UUID
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.vectorstores import FAISS
from langchain.storage import LocalFileStore
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
import streamlit as st
from langchain.callbacks.base import BaseCallbackHandler
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains import LLMChain
from operator import itemgetter

st.set_page_config(
    page_title = "Document GPT",
    page_icon = "📄"
)

@st.cache_data(show_spinner="Embedding file..")
def embed_file(file):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, mode = "wb") as f:
        f.write(file_content)
    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
    
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator = "\n",
        chunk_size = 600,
        chunk_overlap = 100,
    )
    
    loader = UnstructuredFileLoader(file_path)
    
    docs = loader.load_and_split(text_splitter = splitter)
    
    embeddings = OpenAIEmbeddings()
    
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    
    vectorstore = FAISS.from_documents(docs, cached_embeddings)

    retriever = vectorstore.as_retriever()
    
    return retriever

def send_message(message, role, save = True):
    with st.chat_message(role):
        st.markdown(message)
    if(save == True):
        save_message(message, role)
        
def paint_history():
    for message in st.session_state["messages"]:
        send_message(message["message"],message["role"], save = False)
    
def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)

def save_message(message,role):
    st.session_state["messages"].append({"message" : message, "role" : role})

def save_memory(input, output):
    st.session_state["memory"].save_context({"input" : input},{"output" : output})


class ChatCallbackHandler(BaseCallbackHandler):
    
    message = ""
    
    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()
    
    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")
    
    def on_llm_new_token(self, token: str, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)
        
        
llm = ChatOpenAI(
    model = "gpt-4o",
    temperature = 0.1, 
    streaming = True,
    callbacks =[
        ChatCallbackHandler(),
    ])

memory_llm = ChatOpenAI(temperature= 0.1)

template = ChatPromptTemplate.from_messages([
    ("system", """
     You are a helpful AI that analyze the document and give an information to user from that document only.
     Greet the user and be converstional.
     Answer the question using ONLY the following context. If you don't know the answer,
     say you don't know. DO NOT MAKE ANYTHING UP. If the content is not in the document, say it is not in the given document.
     
     Context: {context}
     
     Below is the chat history for you to talk like a conversational AI.
     Use the contents below only if you need to talk about it or the user asked.
     """),
    MessagesPlaceholder(variable_name = "history"),
    ("human", "{question}")
])        
        

if("messages" not in st.session_state):
    st.session_state["messages"] = []
    
if("memory" not in st.session_state):
    st.session_state["memory"] = ConversationSummaryBufferMemory(
        llm=memory_llm, max_token_limit=500, return_messages=True
    )
    
st.title("Document GPT")

st.markdown("""
Welcome!

I learn the document you provide and can answer based on the content.
                
Get started by uploading a file in the sidebar.
""")
with st.sidebar:
    file = st.file_uploader("Upload a .txt .pdf or .docx file", type = ["pdf","txt","docx"])

if(file is not None):
    retriever = embed_file(file)
    send_message("I'm ready! Ask anything", "ai", save = False)
    paint_history()
    
    message = st.chat_input("Ask anything about your file...")

    if(message is not None):
        send_message(message, "human")
        chain = {
                    "context" : retriever | RunnableLambda(format_docs),
                    "question" : RunnablePassthrough(),
                }| RunnablePassthrough.assign(
                    history = RunnableLambda(
                        st.session_state.memory.load_memory_variables
                    )
                    | itemgetter("history")
                )| template | llm
                
        with st.chat_message("ai"):
            response = chain.invoke(message)
            save_memory(message, response.content)

else:
    st.session_state["messages"] = []
    st.session_state["memory"] = ConversationSummaryBufferMemory(
        llm=memory_llm, max_token_limit=500, return_messages=True
    )

