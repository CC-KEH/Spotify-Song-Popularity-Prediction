from pymongo.mongo_client import MongoClient
import os,json
import pandas as pd
from src.constant import uri,DATABASE_COLLECTION_NAME,DATABASE_NAME

# Create a new client and connect to the server
client = MongoClient(uri)

# Send a ping to confirm a successful connection
data = pd.read_csv('notebooks/data/dataset.csv')
data_jsoned = list(json.loads(data.T.to_json()).values())
client[DATABASE_NAME][DATABASE_COLLECTION_NAME].insert_many(data_jsoned)