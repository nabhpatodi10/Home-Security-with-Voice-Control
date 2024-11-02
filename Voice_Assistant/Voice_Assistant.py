import os
from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import MongoDBAtlasVectorSearch
from langchain_community.document_loaders import TextLoader

from langchainget.query import query, get_conversational_chain, get_storage_chain
from db.db import connect_to_mongo

import pyttsx3
import speech_recognition as sr

collection = connect_to_mongo()

recognizer = sr.Recognizer()
tts_engine = pyttsx3.init()

def listen_to_user():
    with sr.Microphone() as source:
        print("Listening...")
        audio = recognizer.listen(source)
        try:
            user_input = recognizer.recognize_google(audio)
            print(f"User said: {user_input}")
            return user_input
        except sr.UnknownValueError:
            print("Sorry, I could not understand the audio.")
            return None
        except sr.RequestError as e:
            print(f"Could not request results; {e}")
            return None
        
def speak_text(text):
    tts_engine.say(text)
    tts_engine.runAndWait()

def voice_assistant():
    #user_question = listen_to_user()
    user_question = """Give me an introduction about myself."""

    if user_question:
        docs = query(collection, user_question)

        if len(docs) <= 0 :
            docs = ""

        conversation_chain = get_conversational_chain()
        response = conversation_chain({"input_documents" : docs, "question" : user_question}, return_only_outputs=True)

        print(f"LLM Response: {response}")
        
        storage_chain = get_storage_chain()
        storage_response = storage_chain({"input_documents" : docs, "question" : user_question, "answer" : response}, return_only_outputs=True)

        print(f"Storage Chain Result: {storage_response}")

        with open("Content.txt", "w") as file:
            file.write(storage_response["output_text"])
        file.close()

        loader = TextLoader("Content.txt")
        docs = loader.load()

        text_spiliter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=20)
        splits = text_spiliter.split_documents(docs)
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key=os.getenv('GEMINI_API_KEY'))
        store = [] 
        for split in splits:
            store.append(split.page_content)

        collection.delete_many({})
            
        docsearch = MongoDBAtlasVectorSearch.from_documents(splits, embeddings, collection=collection, index_name="embeddings")

        speak_text(response["output_text"])


'''if __name__ == "__main__":
    main()'''