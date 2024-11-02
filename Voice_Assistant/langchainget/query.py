import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import MongoDBAtlasVectorSearch
from langchain_groq import ChatGroq
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

def query(collection, query):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=os.getenv("GOOGLE_API_KEY"))
    vectorStore = MongoDBAtlasVectorSearch(collection, embeddings, index_name="vector_index")
    try:
        print("inside the function")
        print(query)
        docs = vectorStore.similarity_search(query, K=4)
        print("Vector Search Results:")
        print(len(docs))
        return docs
    except Exception as e:
        print("Database timeout or error:", str(e))

def get_storage_chain():
    prompt_template = """
    You are a Personal Voice Assistant specifically made for your user, go through all the data that you have, that data is the information that is stored
    about your user. Understand the question asked by the user and the answer which was given to the user. Analyse the data you have on the user and the
    new information which you got from this new conversation. Frame the new information in such a way that anyone can refer to this text, along with the
    other existing data and get to know that person.\n\n
    Context:\n {context}?\n
    Question: \n{question}\n
    Answer: \n{answer}
    """

    model = ChatGroq(model = "llama-3.1-70b-versatile", api_key=os.getenv("GROQ_API_KEY"))

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question", "answer"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain


def get_conversational_chain():
    prompt_template = """
    You are a Personal Voice Assistant specifically made for your user, go through all the data that you have, that data is the information that is stored
    about your user. Understand the question asked by the user and answer with the knowledge you have about your user and then think about what you know
    about the question asked.\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGroq(model = "llama-3.1-70b-versatile", api_key=os.getenv("GROQ_API_KEY"))

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain