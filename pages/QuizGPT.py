from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
import streamlit as st
from langchain.storage import LocalFileStore
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains import LLMChain
from langchain.retrievers import WikipediaRetriever
from langchain.schema import BaseOutputParser
import json

generate_quiz = {
    "name" : "generate_quiz",
    "description" : "function that takes a list of questions and answers and returns a quiz",
    "parameters" : {
        "type" : "object",
        "properties" : {
            "questions" : {
                "type" : "array",
                "items" : {
                    "type" : "object",
                    "properties" : {
                        "question" : {
                            "type" : "string",
                        },
                        "answers" : {
                            "type" : "array",
                            "items" : {
                                "type" : "object",
                                "properties" : {
                                    "answer" : {
                                        "type": "string",
                                    },
                                    "correct" : {
                                        "type": "boolean",
                                    },
                                },
                                "required" : ["answer", "correct"],
                            },
                        },
                    },
                    "required" : ["question", "answers"],
                },
            }
        },
        "required" : ["questions"],
    },
}


st.set_page_config(
    page_title = "Quiz GPT",
    page_icon = "📄"
)

llm = ChatOpenAI(
    temperature = 0.1,
    model = "gpt-4o",
    streaming = True,
    callbacks = [
        StreamingStdOutCallbackHandler()
    ]
)

quiz_llm = ChatOpenAI(
    temperature = 0.1
).bind(
    function_call = {
        "name" : "generate_quiz"
    },
    functions = [
        generate_quiz
    ]
)
def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)

question_prompt = ChatPromptTemplate.from_messages(
    [(
    "system",
    """
        You are a helpful assistant that is role playing as a teacher.
            
        Based ONLY on the following context make 10 questions to test the user's knowledge about the text.
        
        Each question should have 4 answers, three of them must be incorrect and one should be correct.
            
        Use (o) to signal the correct answer.
            
        Question examples:
            
        Question: What is the color of the ocean?
        Answers: Red|Yellow|Green|Blue(o)
            
        Question: What is the capital or Georgia?
        Answers: Baku|Tbilisi(o)|Manila|Beirut
            
        Question: When was Avatar released?
        Answers: 2007|2001|2009(o)|1998
            
        Question: Who was Julius Caesar?
        Answers: A Roman Emperor(o)|Painter|Actor|Model
            
        You:
            
        Context: {context}
    """,
    )
    ]
)

question_chain = {"context" : format_docs} | question_prompt | llm

formatting_prompt = ChatPromptTemplate.from_messages([
    """
    Make a quiz from the following context:
    {context}
    """
])


formatting_chain = formatting_prompt | quiz_llm

@st.cache_data(show_spinner = "Retrieving file...")
def retrieve_file(file):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, mode = "wb") as f:
        f.write(file_content)
    cache_dir = LocalFileStore(f"./.cache/quiz/{file.name}")
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator = "\n",
        chunk_size = 600,
        chunk_overlap = 100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter = splitter)
    
    return docs

@st.cache_data(show_spinner = "Generating quiz...")
def create_quiz_chain(_docs, topic):
    chain = {"context" : question_chain} | formatting_chain
    response = chain.invoke(_docs)
    response = response.additional_kwargs["function_call"]["arguments"]
    return json.loads(response)


@st.cache_data(show_spinner = "Searching Wikipedia...")
def search_wiki(term):
    retriever = WikipediaRetriever(top_k_results = 5)
    docs = retriever.get_relevant_documents(term)
    return docs

st.title("Quiz GPT")

with st.sidebar:
    docs = None
    topic = None
    choice = st.selectbox(
        "Choose the source file to generate Quiz",
        (
            "File",
            "Wikipedia Article",
        ),
    )
    if(choice == "File"):
        file = st.file_uploader(
            "Upload a .docx , .txt or .pdf file",
            type = ["pdf", "txt", "docx"],
        )
        if(file is not None):
            docs = retrieve_file(file)
    else:
        topic = st.text_input("Search Wikipedia...")
        if(topic is not None):
            with st.status("Searching Wikipedia..."):
                docs = search_wiki(topic)

if(docs is None):
    st.markdown(
        """
    I generate based on Wikipedia articles or files you upload.
                
    Get started by uploading a file or searching on Wikipedia in the sidebar.
    """
    )
else:
    response = create_quiz_chain(docs, topic if topic else file.name)
    show_answer = st.toggle("Show the answers with the feedback")
    with st.form("question_form"):
        for question in response["questions"]:
            st.write(question["question"])
            res = st.radio("Select answer", [answer["answer"] for answer in question["answers"]], index = None)
            if({"answer" : res, "correct" : True} in question["answers"]):
                st.success("Correct!")
            elif(res is not None):
                if(show_answer is not None):
                    for answer in question["answers"]:
                        if(answer["correct"] == True):
                            st.error(f"Wrong! Correct answer: { answer['answer'] }")
                else:
                    st.error("Wrong!")
                    
        button = st.form_submit_button()