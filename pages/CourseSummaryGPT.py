from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings
from langchain.storage import LocalFileStore
from langchain.memory import ConversationSummaryBufferMemory
from operator import itemgetter
from pages.DocumentGPT import send_message, paint_history, format_docs, save_memory, save_message
import pages.DocumentGPT as docuGPT
import streamlit as st
import subprocess
import math
import os
from pydub import AudioSegment
import openai
import glob

st.set_page_config(
    page_title = "Course Summary GPT",
    page_icon = "📄"
)

        
llm = ChatOpenAI(temperature= 0.1, streaming= True,
    callbacks=[
        docuGPT.ChatCallbackHandler(),
    ])

summary_llm = ChatOpenAI(temperature= 0.1, streaming= True)

memory_llm = ChatOpenAI(temperature= 0.1)

st.markdown(
    """ 
    # Course Summary GPT
    
    Welcome to Course Summary GPT.
    
    Upload a Video or Audio file and I will give you:
    1. Transcript
    2. Summary
    3. Chat bot to ask any question about the file
    
    Get started with uploading a video/audio file in the sidebar.
    """
)
@st.cache_data()
def extract_audio_from_video(video_path):
    audio_path = video_path.replace("mp4", "mp3")
    command = ["ffmpeg","-y", "-i", video_path, "-vn", audio_path]
    subprocess.run(command)

@st.cache_data()
def cut_audio_in_chunks(audio_path, chunk_size, chunks_folder):
    track = AudioSegment.from_mp3(audio_path)
    chunk_size = chunk_size * 60 * 1000
    chunks = math.ceil(len(track) / chunk_size)
    for i in range(chunks):
        start_time = i * chunk_size
        end_time = (i+1) * chunk_size
        chunk = track[start_time:end_time]
        chunk.export(f"./.cache/files/{chunks_folder}/chunk_{i}.mp3", format = "mp3")
    return chunks

@st.cache_data()
def transcript_chunks(chunks_folder, destination):
    audio_files = glob.glob(f"{chunks_folder}/*.mp3")
    audio_files.sort()
    for audio_file in audio_files:
        with open(audio_file, "rb") as file, open(destination, "a") as text_file:
            transcript = openai.Audio.transcribe("whisper-1", file)
            text_file.write(transcript["text"])

    return 

@st.cache_data()
def embed_file(file_path):
    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
    loader = TextLoader(file_path)
    docs = loader.load_and_split(text_splitter = splitter)
    embeddings = OpenAIEmbeddings()
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever


template = ChatPromptTemplate.from_messages([
    ("system", """
     You are a helpful AI that analyze the document and give an information to user from that document only.
     Greet the user and be converstional. Given context is a transcript of a video or audio file.
     Answer the question using ONLY the following context. If you don't know the answer,
     say you don't know. DO NOT MAKE ANYTHING UP. If the content is not in the document, say it is not in the given document.
     
     Context: {context}
     
     Below is the chat history for you to talk like a conversational AI.
     Use the contents below only if you need to talk about it or the user asked.
     """),
    MessagesPlaceholder(variable_name = "history"),
    ("human", "{question}")
])    

splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size = 800,
    chunk_overlap = 100,
)

if("messages" not in st.session_state):
    st.session_state["messages"] = []
if("memory" not in st.session_state):
    st.session_state["memory"] = ConversationSummaryBufferMemory(
        llm=memory_llm, max_token_limit=500, return_messages=True
    )
    
with st.sidebar:
    video = st.file_uploader(
        "Video/Audio", 
        type = ["mp4", "avi", "mov", "mkv", "mp3"]
        )
    
if(video is not None):
    with st.status("Loading video...") as status:
        video_path = f"./.cache/files/vidoraudio/{video.name}"
        audio_path = video_path.replace("mp4", "mp3")
        
        with open(video_path, "wb") as file:
            file.write(video.read())
            
        status.update(label = "Extracting audio")
        
        extract_audio_from_video(video_path)
        
        status.update(label = "Cutting audio into chunks")
        
        chunks = cut_audio_in_chunks(audio_path, 10, "chunks")
        
        status.update(label = "Transcribing audio...")

        chunks_folder = "./.cache/files/chunks"
        
        transcript_path = video_path.replace("mp4", "txt")
        
        transcript_chunks(chunks_folder, transcript_path)
        
        status.update(label = "Deleting used files...")

        try:
            os.remove(video_path)
            os.remove(audio_path)
            for i in range(chunks):
                os.remove(f"./.cache/files/chunks/chunk_{i}.mp3")
        
        except:
            pass
            
        status.update(label = "Done!")
        
    qna_tab, summary_tab, transcript_tab = st.tabs(["QnA", "Summary", "Transcript"])
    
    with qna_tab:
        send_message("I'm ready! Ask anything!", "ai", save = False)
            
    with summary_tab:
        st.title("Summary of the given file:")
        generate = st.button("Generate summary")
        
        if(generate == True):
            text_loader = TextLoader(transcript_path)
            
            docs = text_loader.load_and_split(text_splitter = splitter)
            
            first_summary_prompt = ChatPromptTemplate.from_template(
                """
                Write a concise summary of the following:
                "{text}"
                CONCISE SUMMARY:                
            """
            )
            
            first_summary_chain = first_summary_prompt | summary_llm
            
            summary = first_summary_chain.invoke({
                "text":docs[0].page_content
            }).content
            
            refine_prompt = ChatPromptTemplate.from_template(
                 """
                Your job is to produce a final summary with explaining to user.
                We have provided an existing summary up to a certain point: {existing_summary}
                We have the opportunity to refine the existing summary (only if needed) with some more context below.
                ------------
                {context}
                ------------
                Given the new context, refine the original summary.
                If the context isn't useful, RETURN the original summary again.
                If the exisiting summary does not need to be refined, RETURN the original summary again.
                """
            )
            
            refine_chain = refine_prompt | summary_llm
            
            with st.status("Summarizing...") as status:
                for index, doc in enumerate(docs[1:]):
                    status.update(label = f"Processing document... {index + 1}/{len(docs) - 1}")
                    summary = refine_chain.invoke({
                        "existing_summary": summary,
                        "context":doc.page_content
                    }).content
            
            st.write(summary)
            
    with transcript_tab:
        st.title("Transcript of the given file:")
        with open(transcript_path, "r") as file:
            st.write(file.read())

    retriever = embed_file(transcript_path)
   
    paint_history()
    message = st.chat_input("Ask anything about your file...")

    if(message is not None):
        send_message(message, "human")
        chain = {
                    "context": retriever | RunnableLambda(format_docs),
                    "question": RunnablePassthrough(),
                }| RunnablePassthrough.assign(
                    history=RunnableLambda(
                        st.session_state.memory.load_memory_variables
                    )
                    | itemgetter("history")
                )| template | llm
        with st.chat_message("ai"):
            response = chain.invoke(message)
            save_memory(message, response.content)