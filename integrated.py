# import streamlit as st
# from PIL import Image
# from PyPDF2 import PdfReader
# from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
# import google.generativeai as genai
# import os
# import re
# import time
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
# from langchain.prompts import PromptTemplate
# from langchain_community.vectorstores import FAISS
# from langchain.chains.question_answering import load_qa_chain
# from google.cloud import aiplatform


# # Load environment variables
# from dotenv import load_dotenv
# load_dotenv()

# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# # Function to load Gemini pro vision model and get responses for image input
# def get_gemini_response_image(input, image):
#     model = genai.GenerativeModel("gemini-1.5-flash")
#     if input:
#         response = model.generate_content([input, image])
#     else:
#         response = model.generate_content(image)
#     return response.text

# # Function to load Gemini pro model for conversational chat
# def get_gemini_chat_response(question):
#     model = genai.GenerativeModel("gemini-1.5-pro")
#     chat = model.start_chat(history=[])
#     response = chat.send_message(question, stream=True)
#     return response

# # Function to extract transcript details from YouTube video
# def extract_video_id(url):
#     pattern = r"(?<=v=)[a-zA-Z0-9_-]+(?=&|\|?|$)"
#     match = re.search(pattern, url)
#     if match:
#         return match.group(0)
#     else:
#         st.error("Invalid YouTube URL")

# def extract_transcript_details(video_id):
#     try:
#         transcript = YouTubeTranscriptApi.get_transcript(video_id)
#         transcript_text = ""
#         for i in transcript:
#             transcript_text += " " + i["text"]
#         return transcript_text
#     except TranscriptsDisabled:
#         st.error("Transcripts are disabled for this video.")
#     except NoTranscriptFound:
#         st.error("No transcript found for this video.")
#     except Exception as e:
#         st.error(f"An error occurred: {e}")

# # Function to generate summary based on prompt from Google Gemini Pro
# def generate_gemini_content(transcript_text, prompt):
#     model = genai.GenerativeModel("gemini-pro")
#     response = model.generate_content(prompt + transcript_text)
#     return response.text

# # Function to read PDF and extract text
# def get_pdf(pdf_docs):
#     text = ""
#     for pdf in pdf_docs:
#         pdf_reader = PdfReader(pdf)
#         for page in pdf_reader.pages:
#             text += page.extract_text()
#     return text

# # Function to split text into chunks
# def get_text_chunks(text):
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
#     chunks = text_splitter.split_text(text)
#     return chunks

# def get_vector_store(text_chunks):
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
#     vector_store.save_local("faiss-index")

# def get_conversational_chain():
#     prompt_template = """
#     Answer the question as detailed as possible from the provided context, make sure to provide all the details, don't provide the wrong answer\n\n
#     Context:\n{context}\n
#     Question:\n{question}\n
#     Answer:
#     """
#     model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
#     prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
#     chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
#     return chain

# def user_input(user_question):
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     new_db = FAISS.load_local("faiss-index", embeddings, allow_dangerous_deserialization=True)
#     docs = new_db.similarity_search(user_question)
#     chain = get_conversational_chain()
#     response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
#     st.write("Reply:", response["output_text"])

# def get_gemini_response(input, image, prompt):
#     model = genai.GenerativeModel('gemini-1.5-flash')
#     response = model.generate_content([input, image[0], prompt])
#     return response.text

# def input_image_details(uploaded_file):
#     if uploaded_file is not None:
#         bytes_data = uploaded_file.getvalue()
#         image_parts = [{"mime_type": uploaded_file.type, "data": bytes_data}]
#         return image_parts
#     else:
#         raise FileNotFoundError("No file uploaded")

# # Retry mechanism decorator
# def retry(max_retries=3, delay=2):
#     def decorator(func):
#         def wrapper(*args, **kwargs):
#             for attempt in range(max_retries):
#                 try:
#                     return func(*args, **kwargs)
#                 except Exception as e:
#                     if attempt < max_retries - 1:
#                         time.sleep(delay)
#                     else:
#                         raise e
#         return wrapper
#     return decorator


# # Main function
# def main():
#     st.set_page_config(page_title="Integrated Streamlit App")
#     st.sidebar.title("Menu")

#     selected_option = st.sidebar.selectbox("Select Project", ["Image to Text", "ConvoLog", "PDF Chat", "Transcriber"])

#     if selected_option == "Image to Text":
#         st.title("Gemini Application")
#         input_text = st.text_input("Input Prompt:", key="input_image")
#         uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
#         image = ""
#         if uploaded_file is not None:
#             image = Image.open(uploaded_file)
#             st.image(image, caption="Uploaded Image", use_column_width=True)
#         submit = st.button("Tell me about the image")
#         if submit and uploaded_file:
#             image_data = input_image_details(uploaded_file)
#             response = get_gemini_response(input_text, image_data, input_text)
#             st.subheader("The Response is")
#             st.write(response)

#     elif selected_option == "ConvoLog":
#         st.title("Gemini LLM Application")
#         if 'chat_history' not in st.session_state:
#             st.session_state['chat_history'] = []
#         input_text = st.text_input("Input:", key="input_convo")
#         submit = st.button("Ask the question")
#         if submit or input_text:
#             response = get_gemini_chat_response(input_text)
#             st.session_state['chat_history'].append(("You", input_text))
#             st.subheader("The Response is")
#             for chunk in response:
#                 st.write(chunk.text)
#                 st.session_state['chat_history'].append(("Bot", chunk.text))
#         st.subheader("The chat history is")
#         if 'chat_history' in st.session_state:
#             for role, text in st.session_state['chat_history']:
#                 st.write(f"{role}: {text}")

#     elif selected_option == "PDF Chat":
#         st.title("Chat with PDF using Gemini")
#         user_question = st.text_input("Ask a Question from the PDF files")
#         if user_question:
#             user_input(user_question)
#         with st.sidebar:
#             st.title("Menu")
#             pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
#             if st.button("Submit & Process"):
#                 with st.spinner("Processing..."):
#                     raw_text = get_pdf(pdf_docs)
#                     text_chunks = get_text_chunks(raw_text)
#                     get_vector_store(text_chunks)
#                     st.success("Done")

    # elif selected_option == "Invoice Parsing":
    #     st.title("Multilanguage Invoice Extractor")
    #     input_text = st.text_input("Input Prompt:", key="input_invoice")
    #     uploaded_file = st.file_uploader("Choose the image of Invoice...", type=["jpg", "jpeg", "png"])
    #     if uploaded_file is not None:
    #         image = Image.open(uploaded_file)
    #         st.image(image, caption="Uploaded Image", use_column_width=True)
    #     submit = st.button("Tell me about the invoice")
    #     input_prompt = "You are an expert in understanding invoices. We will upload an image as an invoice and you will have to answer any questions based on the uploaded invoice."
    #     if submit and uploaded_file:
    #         image_data = input_image_details(uploaded_file)
    #         response = get_gemini_response(input_text, image_data, input_prompt)
    #         st.subheader("The Response is")
    #         st.write(response)
    
  
#     elif selected_option == "Transcriber":
#         st.title("YouTube transcript to detailed notes converter")
#         prompt = "YouTube video summarizer, taking the transcript text and summarize the entire video and providing the important summary in points within 250 words. Please provide the summary of the text given here: "
#         youtube_link = st.text_input("Enter YouTube video link")
#         if youtube_link:
#             video_id = extract_video_id(youtube_link)
#             st.image(f"http://img.youtube.com/vi/{video_id}/0.jpg")
#         submit = st.button("Generate Summary")
#         if submit and youtube_link:
#             with st.spinner("Generating Transcript..."):
#                 transcript_text = extract_transcript_details(video_id)
#                 if transcript_text:
#                     st.subheader("The transcript is")
#                     st.write(transcript_text)
#                     st.subheader("The summary is")
#                     content = generate_gemini_content(transcript_text, prompt)
#                     st.write(content)

# if __name__ == "__main__":
#     main() 

import streamlit as st
from PIL import Image
from PyPDF2 import PdfReader
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
import google.generativeai as genai
import os
import re
import time
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from google.cloud import aiplatform
import io
# Load environment variables
from dotenv import load_dotenv
load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# ✅ Function to process image input using Gemini
def get_gemini_response_image(input, image):
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content([input, image] if input else image)
    return response.text

# ✅ Function for Conversational Chat with Gemini
def get_gemini_chat_response(question):
    model = genai.GenerativeModel("gemini-1.5-pro")
    chat = model.start_chat(history=[])
    response = chat.send_message(question, stream=True)
    return response

# ✅ Function to extract YouTube transcript
def extract_video_id(url):
    match = re.search(r"(?<=v=)[a-zA-Z0-9_-]+(?=&|\?|$)", url)
    return match.group(0) if match else None

def extract_transcript_details(video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join([i["text"] for i in transcript])
    except TranscriptsDisabled:
        return "Transcripts are disabled for this video."
    except NoTranscriptFound:
        return "No transcript found for this video."
    except Exception as e:
        return f"Error: {e}"

# ✅ Function to summarize transcripts
def generate_gemini_content(transcript_text, prompt):
    model = genai.GenerativeModel("gemini-1.5-pro")
    response = model.generate_content(prompt + transcript_text)
    return response.text

# ✅ Function to extract text from PDFs
def get_pdf(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text() or ""  # Handle NoneType errors
            text += page_text
    return text

# ✅ Function to split text into manageable chunks
def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return splitter.split_text(text)

# ✅ Function to create vector embeddings for retrieval
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss-index")

# ✅ Function to create conversational chain for QA
def get_conversational_chain():
    prompt_template = """
    Answer the question based on the given context. Provide detailed, accurate responses.
    \n\nContext:\n{context}\n
    \nQuestion:\n{question}\n
    \nAnswer:
    """
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0.3)
    retriever = FAISS.load_local("faiss-index", GoogleGenerativeAIEmbeddings(model="models/embedding-001"), allow_dangerous_deserialization=True).as_retriever()

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)
    
    return qa_chain

# ✅ Function to process user queries on PDFs
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.load_local("faiss-index", embeddings, allow_dangerous_deserialization=True)
    docs = vector_store.similarity_search(user_question)

    chain = get_conversational_chain()
    response = chain.invoke({"input_documents": docs, "query": user_question})
    
    st.write("Reply:", response["result"])

# ✅ Function to process uploaded images
def input_image_details(uploaded_file):
    if uploaded_file:
        return [{"mime_type": uploaded_file.type, "data": uploaded_file.getvalue()}]
    raise FileNotFoundError("No file uploaded")

# ✅ Retry mechanism decorator
def retry(max_retries=3, delay=2):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt < max_retries - 1:
                        time.sleep(delay)
                    else:
                        raise e
        return wrapper
    return decorator

# ✅ Main function: Streamlit UI
def main():
    st.set_page_config(page_title="AI-Powered Streamlit App")
    st.sidebar.title("Menu")

    selected_option = st.sidebar.selectbox("Select Feature", ["Image to Text", "ConvoLog", "PDF Chat", "YouTube Transcriber"])
    # if selected_option == "Image to Text":
    #     st.title("Gemini Application")
    #     input_text = st.text_input("Input Prompt:", key="input_image")
    #     uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    #     image = ""
    #     if uploaded_file is not None:
    #         image = Image.open(uploaded_file)
    #         st.image(image, caption="Uploaded Image", use_column_width=True)
    #     submit = st.button("Tell me about the image")
    #     if submit and uploaded_file:
    #         image_data = input_image_details(uploaded_file)
    #         response = get_gemini_response_image(input_text, image_data)
    #         st.subheader("The Response is")
    #         st.write(response)


    # if selected_option == "Image to Text":
    #     st.title("Image Analysis using Gemini")
    #     input_text = st.text_input("Enter Prompt:")
    #     uploaded_file = st.file_uploadeimport io
    from PIL import Image
    if selected_option == "Image to Text":
        st.title("Gemini Application")
    
        input_text = st.text_input("Input Prompt:", key="input_image")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
        image = None  # Initialize image variable
        response = None  # Initialize response variable

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)

        submit = st.button("Tell me about the image")

        if submit and uploaded_file:
            image_data = input_image_details(uploaded_file)  # Get image details

        # Extract image data correctly
            if isinstance(image_data, list) and len(image_data) > 0 and 'data' in image_data[0]:
                image_bytes = image_data[0]['data']  # Extract binary image data
                image = Image.open(io.BytesIO(image_bytes))  # Convert to PIL Image

            if image:
                response = get_gemini_response_image(input_text, image)  # Call function with fixed parameters
                if response:  # Ensure response exists before displaying
                    st.subheader("The Response is")
                    st.write(response)
                else:
                    st.error("Failed to generate a response.")
            else:
                st.error("No valid image found.")
# r("Upload Image", type=["jpg", "jpeg", "png"])
        
    #     if uploaded_file:
    #         st.image(Image.open(uploaded_file), caption="Uploaded Image", use_column_width=True)
        
    #     if st.button("Analyze Image") and uploaded_file:
    #         response = get_gemini_response_image(input_text, input_image_details(uploaded_file))
    #         st.write("Response:", response)

    # elif selected_option == "Conversational Chat":
    #     st.title("Conversational AI Chatbot")
    #     user_input_text = st.text_input("Ask a question:")
        
    #     if st.button("Ask") or user_input_text:
    #         response = get_gemini_chat_response(user_input_text)
    #         for chunk in response:
    #             st.write(chunk.text)

    elif selected_option == "ConvoLog":
        st.title("Gemini LLM Application")
        if 'chat_history' not in st.session_state:
            st.session_state['chat_history'] = []
        input_text = st.text_input("Input:", key="input_convo")
        submit = st.button("Ask the question")
        if submit or input_text:
            response = get_gemini_chat_response(input_text)
            st.session_state['chat_history'].append(("You", input_text))
               
            st.subheader("The Response is")
            for chunk in response:
                st.write(chunk.text)
                st.session_state['chat_history'].append(("Bot", chunk.text))
            st.subheader("The chat history is")
            if 'chat_history' in st.session_state:
                for role, text in st.session_state['chat_history']:
                    st.write(f"{role}: {text}")



    elif selected_option == "PDF Chat":
        st.title("Chat with PDFs")
        user_question = st.text_input("Ask a question about uploaded PDFs")

        with st.sidebar:
            pdf_docs = st.file_uploader("Upload PDF files", accept_multiple_files=True)
            if st.button("Process PDFs"):
                with st.spinner("Processing..."):
                    text_chunks = get_text_chunks(get_pdf(pdf_docs))
                    get_vector_store(text_chunks)
                    st.success("Processing Complete")

        if user_question:
            user_input(user_question)

    elif selected_option == "YouTube Transcriber":
        st.title("YouTube Video Transcription")
        prompt = "YouTube video summarizer, taking the transcript text and summarize the entire video and providing the important summary in points within 250 words. Please provide the summary of the text given here: "
        video_url = st.text_input("Enter YouTube URL")
        if video_url:
            video_id = extract_video_id(video_url)
            st.image(f"http://img.youtube.com/vi/{video_id}/0.jpg")
        
        
        if st.button("Get Transcript"):
            video_id = extract_video_id(video_url)
            if video_id:
                transcript = extract_transcript_details(video_id)
                st.write("Transcript:", transcript)

# ✅ Run Streamlit App
if __name__ == "__main__":
    main()


#     elif selected_option == "Transcriber":
#         st.title("YouTube transcript to detailed notes converter")
#         prompt = "YouTube video summarizer, taking the transcript text and summarize the entire video and providing the important summary in points within 250 words. Please provide the summary of the text given here: "
#         youtube_link = st.text_input("Enter YouTube video link")
#         if youtube_link:
#             video_id = extract_video_id(youtube_link)
#             st.image(f"http://img.youtube.com/vi/{video_id}/0.jpg")
#         submit = st.button("Generate Summary")
#         if submit and youtube_link:
#             with st.spinner("Generating Transcript..."):
#                 transcript_text = extract_transcript_details(video_id)
#                 if transcript_text:
#                     st.subheader("The transcript is")
#                     st.write(transcript_text)
#                     st.subheader("The summary is")
#                     content = generate_gemini_content(transcript_text, prompt)
#                     st.write(content)

# if __name__ == "__main__":
#     main() 