from qdrant_client import QdrantClient
from dotenv import load_dotenv
import os
from qdrant_client.models import Distance, VectorParams
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.vectorstores import Qdrant

load_dotenv() # take env variables from .env file

# create the qdrant client for connecting to vector db
qdrant_url = os.getenv("qdrant_url")
print("qdrant_url : ", qdrant_url)
qdrant_client = QdrantClient(url=qdrant_url)
print("Qdrant Client created successfully")

# create the collection
collectionName = "collection_of_text_blobs_with_chunks"
qdrant_client.recreate_collection(
    collection_name=collectionName,
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
)
print("Qdrant collection created successfully")

# START -- data load from directory
loader = DirectoryLoader('./data',glob="./*.txt", show_progress=True)
data = loader.load()
print("Length of data :", len(data))
text = data[0].page_content
print("Data loaded successfully")

# HuggingFace embeddings setup
inference_api_key = os.getenv("hf_token")
model_name = os.getenv("embedding_model_name")
print("Embedding Model Name = ", model_name)
embeddings = HuggingFaceInferenceAPIEmbeddings(
    api_key=inference_api_key, model_name=model_name
)

# Using HuggingFace embeddings with SemanticChunker
# text_splitter = SemanticChunker(embeddings)
# docs = text_splitter.create_documents([text])
# print(docs[0].page_content)

#text_splitter = SemanticChunker(embeddings)
text_splitter = SemanticChunker(
    embeddings, breakpoint_threshold_type="percentile" # "standard_deviation", "interquartile"
)
documents = text_splitter.create_documents([text])
print("Length of documents [No. of chunks] :", len(documents))

# START -- create the vector store
qdrant = Qdrant.from_documents(
    documents,
    embeddings,
    url=qdrant_url,
    collection_name=collectionName,
    force_recreate=True
)
print("Vector store created successfully")