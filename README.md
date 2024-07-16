# Smart College Helper GPT
A comprehensive AI for college students. It includes tools for document analysis, therapy, quiz generation, course information retrieval, and multimedia summarization. Built on Streamlit and specialized GPT models, it enhances academic efficiency and supports student success with advanced computational techniques.

## GPTs

There are five specialized GPTs in this project: 
- [DocumentGPT](#document-gpt)
- [TherapistGPT](#therapist-gpt)
- [DenisonCourseQnA_GPT](#denison-course-qna-gpt)
- [CourseSummaryGPT](#course-summary-gpt)
- [QuizGPT](#quiz-gpt)

All GPTs utilize conversationSummaryBufferMemory to store recent memories and provide summarized versions of older memories, enabling them to understand and remember previous conversations with the user.

## Document GPT
### Analyzes and interprets long documents, saving time by quickly extracting information.

<img width="785" alt="Document1" src="https://github.com/user-attachments/assets/2c10aeff-d881-4df7-a92e-00bf189cd78c">

- **Functionality:**

    - Designed to understand and interpret long documents.
    - Answers questions only based on the document's content.
      
- **Technical Details:**
  
    - Utilizes a map-reduce chain that combines responses from LLM into a final document to provide the best answers to enhance the overall efficiency and accuracy of information retrieval.
    - Embeds the text from the given document.

## Therapist GPT
### A private AI therapist, offering confidential mental health support based on fine-tuned models.

<img width="362" alt="Screenshot" src="https://github.com/user-attachments/assets/4f6b07f9-0195-4e19-890e-9c424b94366b">
<img width="1165" alt="thera35" src="https://github.com/user-attachments/assets/5e1607fb-c5ae-49e4-9efd-ea00288cd5d8">
<img width="1179" alt="thera824" src="https://github.com/user-attachments/assets/5c1a6b7c-af18-438a-af33-eec5f85898da">

- **Functionality:**
  
    - Functions as a private AI therapist ensuring private and efficient interactions for mental well-being support.
    - Users can choose between the local model for privacy or ChatGPT for additional insights.
      
- **Technical Details:**
    - Fine-tuned LLama3 instruct models on extensive therapist conversation datasets (3.5k and 824k rows) for better response.
    - Quantized to GGUF format with q4_k_m method to optimize performance and speed.
 
## Denison Course QnA GPT
### Retrieves current degree requirements from Denison University's website, aiding academic planning with accurate course information.

<img width="790" alt="denison_course1" src="https://github.com/user-attachments/assets/af3d03bc-38da-4d7e-91ea-d813ebe4bbe0">
<img width="747" alt="denison_course2" src="https://github.com/user-attachments/assets/ef219298-efab-4d44-9e58-8d31db64b63c">
<img width="775" alt="denison_course3" src="https://github.com/user-attachments/assets/b57a352b-668f-4d6c-aa66-b27b6c54c636">

 - **Functionality:**
   
    - Retrieves the most up-to-date degree requirement information from the Denison University website.
    - Answers questions about the courses or degree requirements with the information source.
      
- **Technical Details:**
  
    - Uses SitemapLoader to retrieve the latest course information, parses HTML with Beautiful Soup, and stores embedded vectors.
    - Utilizes a Map Re-rank chain in which LLM responds using document information only, scoring them from 0 to 5, selects the highest-rated and most recent answers, and responds to the question from the user including the source URL.
    - Detects similar questions, caches questions and answers, and returns cached responses to avoid duplicated computations.

 ## Course Summary GPT
 ### Creates Q&A ChatBot, summaries, and transcripts of educational videos or lectures, facilitating efficient study sessions with condensed multimedia content or the record of course.

<img width="773" alt="Course Summary GPT" src="https://github.com/user-attachments/assets/7f2b8b92-ef17-4309-811e-946c56f28d1c">
<img width="798" alt="coursesummary1" src="https://github.com/user-attachments/assets/08849c38-3b22-4190-adad-7bfd9e3ddab1">
<img width="794" alt="coursesummary2" src="https://github.com/user-attachments/assets/ed2da7f7-5fe5-4000-b49d-9a39bf22882c">

 - **Functionality:**
   
    - Offers Q&A ChatBot, summaries, and transcript functionalities for given video or audio files.
    - Beneficial for reviewing course materials or any educational videos.
      
- **Technical Details:**
  
    - Parses audio/video files and utilizes the Whisper model to generate accurate transcripts from the recordings.
    - Uses a refine chain to summarize audio/video content by loading the complete transcript in segments, creating and refining summaries progressively until all documents are read.
 
## Quiz GPT
### Generates quizzes from documents or Wikipedia, reinforcing learning through interactive study aids.

<img width="1142" alt="QuizGPT" src="https://github.com/user-attachments/assets/316dd1d1-31e5-4e88-a61c-abc111de508e">

- **Functionality:**
  
    - Generates quizzes based on provided documents or information retrieved from Wikipedia.
    - Ideal for test preparation, allowing users to test their knowledge and reinforce learning.
      
- **Technical Details:**
  
    - Uses a wikipedia retriever to gather information.
    - Utilizes function calling in GPT to format quizzes as intended.
    - Caches files or searched topics to prevent duplicated computations.
