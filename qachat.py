from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import os
import google.generativeai as genai

genai.configure(api_key= os.getenv('GOOGLE_API_KEY'))

#functions to load Gemini pro model as a response 

model= genai.GenerativeModel("gemini-pro")
chat= model.start_chat(history=[])

def get_gemini_response(question):
    response = chat.send_message(question, stream=True)
    return response

#initialize our streamlit app
st.set_page_config(page_title = "Q&A Demo")

st.header("Gemini LLM Application")

#initialize session state for chat history if it doesnt exist 
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

input_text=st.text_input("Input:",key="input_text")
submit=st.button("Ask the question")

if submit and input_text:
    response = get_gemini_response(input_text)
    #Add user querry and response to session chat history
    st.session_state['chat_history'].append(("You",input_text))
    st.subheader("The Response is")
    for chunk in response:
        st.write(chunk.text)
        st.session_state['chat_history'].append(("Bot", chunk.text))
        
st.subheader("The chat history is")

if 'chat_history' in st.session_state:
 for role, text in st.session_state['chat_history']:
    st.write(f"{role}:{text}")