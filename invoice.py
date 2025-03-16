from dotenv import load_dotenv
load_dotenv() #load all env variables from .env file

import streamlit as st
import os 
from PIL import Image 
import google.generativeai as genai

genai.configure(api_key = os.getenv('GOOGLE_API_KEY'))

#functions to lead environment variables 
model = genai.GenerativeModel('gemini-pro-vision')

def get_gemini_response(input,image,prompt):
    response = model.generate_content([input, image[0], prompt])
    return response.text
##here we have three parameters 1. whatever input i really want with respect to the images im giving 
## image going to pass , prompt for the address

#when submit button is cicked again we have to load the image and do 
#some processing 
#take the uploaded file and convert it into some bytes

def input_image_details(uploaded_file):
    if uploaded_file is not None:
        #Read the file into bytes, by converting it into bytes
        bytes_data = uploaded_file.getvalue()

        image_parts = [
            {
                "mime_type": uploaded_file.type, #Get the mime type of the up
                "data": bytes_data
            }
        ]
        return image_parts
    else:
        raise FileNotFoundError("No file uploaded")


#initialize our streamlit app

st.set_page_config(page_title = "Multilanguage Invoice Extractor")
st.header("Gemini Application")

input = st.text_input("Input Prompt: ", key="input")
uploaded_file = st.file_uploader("Choose the image of Invoice...", type= ["jpg", "jpeg", "png"])
image=""
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption= "Uploaded Image", use_column_width = True)


submit = st.button("Tell me about the invoice")

input_prompt = """
You are an expert in understanding envoices. We will upload a image as a invoice 
and you will have to answer any questions based on the iploaded image envoice 
"""

#if submit button is clicked 
if submit:
    image_data = input_image_details(uploaded_file)
    response = get_gemini_response(input, image_data,input_prompt)

    st.subheader("The Response is")
    st.write(response)