from langchain.document_loaders import SitemapLoader
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.storage import LocalFileStore
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
import streamlit as st
from langchain.memory import ConversationSummaryBufferMemory
from operator import itemgetter
import pages.DocumentGPT as docuGPT
from pages.DocumentGPT import send_message, paint_history, save_memory, save_message

st.set_page_config(
    page_title="DenisonCourseQA",
    page_icon="🖥️",
)

answer_llm = ChatOpenAI(
    model = "gpt-4o",
    temperature = 0.1,
    streaming = True,
    callbacks = [
        docuGPT.ChatCallbackHandler(),
    ]
)
question_llm = ChatOpenAI(
    model = "gpt-4o",
    temperature = 0.1
)

llm = ChatOpenAI(
    model = "gpt-4o",
    temperature = 0.1
)

memory_llm = ChatOpenAI(
    model = "gpt-3.5-turbo",
    temperature = 0.1
)

if("messages" not in st.session_state):
    st.session_state["messages"] = []
    
if("questions" not in st.session_state):    
    st.session_state["questions"] = []
    
if("memory" not in st.session_state):
    st.session_state["memory"] = ConversationSummaryBufferMemory(
        llm = memory_llm, max_token_limit = 500, return_messages = True
    )
    
answers_prompt = ChatPromptTemplate.from_messages([
    ("system","""
    Using ONLY the following context answer the user's question. If you can't just say you don't know, don't make anything up.
                                                  
    Then, give a score to the answer between 0 and 5.

    If the answer answers the user question the score should be high, else it should be low.
    
    Make sure to always include the answer's score even if it's 0.
    
    Use the courses that are only in the following context. DO NOT MAKE UP a course.
    Make sure to give a course requirement list of the specific major the user asked. DO NOT GIVE the course requirement list from other majors.
    Use the contents in the context only. DO NOT MAKE UP.
    
    Use the content in the context as much as possible to answer the user's question.
    Context: {context}

    Examples:

    Question: What is the degree requirement for DA major?
    Answer: The major in Data Analytics (DA) requires a minimum of 46 credits of coursework and an approved summer experience.
    DA 101	Introduction to Data Analytics
    CS 109	Discovering Computer Science
    or CS 111	Discovering Computer Science: Scientific Data and Dynamics
    or CS 112	Discovering Computer Science: Markets, Polls, and Social Networks
    MATH 135	Single Variable Calculus
    or MATH 145	Multi-variable Calculus
    DA 200	Data Analytics Colloquium (once as a sophomore and once as a junior or senior, 2 credits total)
    DA 210/CS 181	Data Systems
    DA/MATH 220	Applied Statistics
    DA 301	Practicum in Data Analytics
    DA 350	Advanced Methods for Data Analytics
    DA 401	Seminar in Data Analytics
    (b)  Second, students must complete a DA summer experience (internship or research project).  This experience must be approved by the Data Analytics Program Committee, and is normally undertaken during the summer before the senior year.
    (c) Third, students must acquire some depth in a domain of Data Analytics.  
    They will then carry this disciplinary knowledge into their summer experience and senior seminar.  Students may satisfy this requirement in one of two ways.  First, they may choose to take the designated set of courses from one of the following departments.
    Courselists...
    Score: 5
    
    Question: What is the degree requirement for DA major?
    Answer: 
    Computer Science offers two degrees, a minor, and a concentration.  The two majors both require the computer science core curriculum.  The core courses in Computer Science are:
    Code	Title
    An introductory course
    CS 109	Discovering Computer Science
    or CS 110	Discovering Computer Science: Digital Media and Games
    or CS 111	Discovering Computer Science: Scientific Data and Dynamics
    or CS 112	Discovering Computer Science: Markets, Polls, and Social Networks
    CS 173	Intermediate Computer Science
    CS 181	Data Systems
    CS 234	Mathematical Foundations of Computer Science
    CS 271	Data Structures
    CS 281	Introduction to Computer Systems
    CS 371	Algorithm Design and Analysis
    CS 395	Technical Communication I
    MATH 135	Single Variable Calculus
    Bachelor of Arts Degree
    The minimum requirements for a Bachelor of Arts degree in Computer Science are the core courses plus two additional Computer Science courses at the 300 or 400 level (excluding 395/495, 361-362 and 363-364). One of the 300 or 400 level electives must be a Systems course and the other must either be a Theory or Applied elective.

    Score: 0 (The User asked for the requirement of DA major but the answer is about CS major.)
                                                  
    Question: I am majoring in CS right now. I took until CS173. What course should I take?
    Answer: CS181 and CS234 will be great courses to take after CS173
    Score: 5
    
    Question: I am majoring in DA right now. I took until DA210. What course should I take?
    Answer: DA320 will be great courses to take after DA210. (DA320 does not exist)
    Score: 0
            
    Question: What is the requirment for bio-quantum computation major?
    Answer: There is no such major at Denison. I don't know
    Score: 0  
    
    Question: What courses do we have to take to major in CS?
    Answer: You have to take CS173, CS234, CS271 and more
    Score: 2 (Not sufficient information)                             
    Your turn!

    Below is the chat history for you to talk like a conversational AI.
    Use the contents below only if you need to talk about it or the user asked.

    History:
"""
),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}")
])

similar_prompt = ChatPromptTemplate.from_template(
    """
    You are the AI the determines if the human asked the similar question. 
    
    From the following context, if there is any question that are simliar or same to the human question, return that question in the exact format.
    DO NOT CHANGE any content of the question. Return that similar question in the original format.
    If there is no similar question, return "False". do not return in other format.
    
    human question: {question}
    
    context: {context}
    
    Your answer:
    """
)

choose_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Use ONLY the following pre-existing answers to answer the user's question.

            Use the answers that have the highest score (more helpful) and favor the most recent ones.

            Cite sources and return the sources of the answers as they are, do not change them.
            Cite sources only if they are relevant and you actually used. Do not site if the source wasn't relevant

            Answers: {answers}
            """,
        ),
        ("human", "{question}"),
    ]
)

def similar_question(q):
    question = q
    condensed = "\n\n".join(question for question in st.session_state["questions"])
    chain = similar_prompt | question_llm
    
    return chain.invoke({"question" : question, "context" : condensed})

def choose_answer(inputs):
    answers = inputs["answers"]
    question = inputs["question"]
    choose_chain = choose_prompt | answer_llm
    condensed = "\n\n".join(
    f"{answer['answer']}\nSource:{answer['source']}\nDate:{answer['date']}\n"
    for answer in answers
    )
    return choose_chain.invoke({
        "question" : question,
        "answers" : condensed
    })

def get_answers(inputs):
    docs = inputs["docs"]
    question = inputs["question"]
    answers_chain = RunnablePassthrough.assign(
                    history=RunnableLambda(
                        st.session_state.memory.load_memory_variables
                    )
                    | itemgetter("history")
                ) | answers_prompt | llm
    res = dict()
    res["question"] = question
    res["answers"] = []
    for doc in docs:
        result = {
            "answer" : answers_chain.invoke(
            {
                "question" : question, 
                "context" : doc.page_content
            }
        ).content,
            "source" : doc.metadata["source"],
            "date": doc.metadata["lastmod"]
        }
        res["answers"].append(result)
    return res

def parse_page(soup):
    section = soup.find("section")
    if(section is not None):
        text = str(section.get_text())
        text = text.replace("\n","")
        return text
    return "none"

@st.cache_data(show_spinner = "Loading website...")
def load_website(url):
    loader = SitemapLoader(
    url,
    filter_urls=[
    r"^(https:\/\/denison.edu\/academics\/).*\/degree-requirements",
    ],
    parsing_function = parse_page
    )
    
    loader.requests_per_second = 0.1
    
    docs = loader.load()
    
    cache_dir = LocalFileStore(f"./.cache/embeddings/{url[:9]}")
    
    embeddings = OpenAIEmbeddings()
    
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings,cache_dir)
    
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    
    return vectorstore.as_retriever()

@st.cache_data(show_spinner = "Searching...")
def get_response(question):
    chain = {
        "docs": retriever,
        "question" : RunnablePassthrough(),
    } | RunnableLambda(get_answers) | RunnableLambda(choose_answer)
    return chain.invoke(question)

st.title("Denison University Course Q&A")
st.markdown(
    """            
    Ask anything about the courses at Denison University.
"""
)

if("messages" not in st.session_state):
    st.session_state["messages"] = []
    
if("questions" not in st.session_state):
    st.session_state["questions"] = []
    
if("memory" not in st.session_state):
    st.session_state["memory"] = ConversationSummaryBufferMemory(
    llm=memory_llm, max_token_limit=500, return_messages=True
)
    
retriever = load_website("https://denison.edu/sitemap.xml")

send_message("I'm ready! Ask anything", "ai", save = False)
paint_history()

user_question = st.chat_input("Ask questions")

if(user_question is not None):
    dup_question = similar_question(user_question).content
    original_question = user_question
    
    if(dup_question != 'False'):
        user_question = dup_question
        
    st.session_state["questions"].append(user_question)
    send_message(original_question, "human")
    
    with st.chat_message("ai"):
        res = get_response(user_question)
        res = res.content.replace("$", "\$")
        if(dup_question != 'False'):
            save_message(res, "ai")
            
        save_memory(original_question, res)
else:
    st.session_state["messages"] = []
    st.session_state["memory"] = ConversationSummaryBufferMemory(
        llm=memory_llm, max_token_limit=500, return_messages=True
    )

