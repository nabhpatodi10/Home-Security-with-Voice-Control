from pymongo import MongoClient
import os

def connect_to_mongo():
    try:
      client = MongoClient(os.getenv("MONGODB_URI"))
      collection = client['LangChain']['vectors']
      print("Connected to DB get set to use it")
      return collection
    except Exception as e:
      print(e)