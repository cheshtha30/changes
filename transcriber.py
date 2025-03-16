#youtube video transcriber
#here we will give the youtube video link and it will generate the text from the videos 

import streamlit as st
from dotenv import load_dotenv
load_dotenv() #load all env variables 

import google.generativeai as genai
import os

from youtube_transcript_api import YouTubeTranscriptApi #get idea of the video from URL and retrieve the entire transcript get all the transcripts 
genai.configure(api_key = os.getenv('GOOGLE_API_KEY'))
#now we will fetch transcripts , which will be able to fetch transcript details 
#first of all take my transcripts this will be a text in short whatever we will take from the video
#secondly we can get subject like what is this transcript all about datascience , physics category wise if dont want to use this u can just take the transcript and tell to provide a summary but all the videos need to be public

prompt= "Youtube video summarizer, taking the transcript text and summarize the entire video and providing the important summary in points within 250 words, please provide the summary of the text given here """

#given this simple prompt, will use the model , will create then take model and generate response
#can also create a pormpt variable

#here we will extract transcript details 

#from youtube url we will make a try catch block to handle url
#firstly we will give video url . Get the youtube video id which we will be able to extract the entire transcript, by simple python code .split
#based on the video it will be divided into 1st index and 2nd index
#these transcripts are in the form of list and will append this one by one to form this paragraph
#1st index has the id 

#getting the transcript data from youtube videos
def extract_transcript_details(youtube_video_url, language_code='en'):
    try:
        video_id = youtube_video_url.split("=")[1]
    
        transcript_text=YouTubeTranscriptApi.get_transcript(video_id, languages=[language_code])

        transcript = ""
        for i in transcript_text:
            transcript += "" +i["text"]
        return transcript

    except Exception as e:
        st.error("Error: Could not recieve transcript.")
        st.error(str(e))
        return None

#getting summary based on prompt from Google Gemini Pro
 
def generate_gemini_content(transcript_text, prompt):
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(prompt+transcript_text)
    return response.text
#since im appending over here which means i get appended text
#this is when we are interacting with the gemini pro method, otheer method is to take out transcript from the video 
#and do summarization of transcript
    
#create streamlit app
#create text input box which will help me get a youtube link
st.title("Youtube transcript to detailed notes converter")
youtube_link = st.text_input("Enter youtube video link")


if youtube_link:
    video_id= youtube_link.split("=")[1] #split with the help of youtube id and we can use the image to display the image itself
    #whenever u try to upload image in form of thumbnail the reference is video id, the image of the perticular detail is  displayed in the bottom 
    st.image(f"http://img.youtube.com/vi/{video_id}/0.jpg", use_column_width=True)

#now will create a button for detailed discription of function
if st.button("Get Detailed Notes"):
    transcript_text=extract_transcript_details(youtube_link)
    #extract transcript from youtube link, will call thhis and will get the entire text
    #if i get the transcript text the next function i have to call is generate gemini summary
    #and will give two important function one is the transcript text and the other is the prompt 

    if transcript_text:
        summary=generate_gemini_content(transcript_text, prompt)
        st.markdown("Detailed Notes")
        st.write(summary)