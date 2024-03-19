from qdrant_client import QdrantClient
from dotenv import load_dotenv
import os
from qdrant_client.models import Distance, VectorParams
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.vectorstores import Qdrant
#from langchain_community.vectorstores.qdrant import QdrantVectorStore
from langchain_community.llms import HuggingFaceHub
from langchain_community.llms import HuggingFaceEndpoint
from langchain.chains import RetrievalQA
import gradio as gr
from gradio.themes.base import Base


load_dotenv() # take env variables from .env file

# create the qdrant client for connecting to vector db
qdrant_url = os.getenv("qdrant_url")
print("qdrant_url : ", qdrant_url)
qdrant_client = QdrantClient(url=qdrant_url)
print("Qdrant Client created successfully")

# create the collection
collectionName = "collection_of_text_blobs_with_chunks"

# HuggingFace embeddings setup
inference_api_key = os.getenv("hf_token")
model_name = os.getenv("embedding_model_name")
print("Embedding Model Name = ", model_name)
embeddings = HuggingFaceInferenceAPIEmbeddings(
    api_key=inference_api_key, model_name=model_name
)

# START -- create the vector store
#vectorStore = QdrantVectorStore.fromExistingCollection( embeddings)
vectorStore = Qdrant(
    embeddings = embeddings,
    client=qdrant_client,
    collection_name=collectionName
#    metadata_payload_key="tags",
#   content_payload_key="content",
)


def query_data(query):
    print("query = ",query)
    ### Search the vector store with the query
    docs = vectorStore.similarity_search(query, k=4)
    print("Length of docs-->",len(docs))
    ### Get the first and the most relevant search result
    as_output = docs[0].page_content
    model_name = os.getenv("llm_model_name")
    print("LLM Model =", model_name)
    #llm = OpenAI(openai_api_key=key_param.openai_api_key,temperature = 0, max_tokens=2048)
    #llm = HuggingFaceHub(huggingfacehub_api_token=inference_api_key,repo_id=model_name,task="text2text-generation",model_kwargs={"temperature":0.7,"max_new_tokens":50,"truncation": "only_first"})
    llm = HuggingFaceEndpoint(repo_id=model_name, max_length=128, temperature=0.5, huggingfacehub_api_token=inference_api_key)
    print("llm created")

    retriever = vectorStore.as_retriever()
    print("retriever created")
    ###chain_type_kwargs = {"prompt": query}

    ###qa = RetrievalQAWithSourcesChain.from_chain_type(llm, chain_type="stuff", retriever = retriever, verbose=True,reduce_k_below_max_tokens=True)

    qa = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever = retriever)
    print("attempting to start the query")
    retriever_output = qa.invoke(query)
    print("answer received")
    return as_output, retriever_output

with gr.Blocks(theme=gr.themes.Glass(),title="Question Answering App using Qdrant Vector Search and RAG (Ask about Medicare advantage plan)") as demo:
    gr.Markdown(
        """
        # Question Answering App using Qdrant Vector Search and RAG (Ask about Medicare advantage plan)
        """)
    textbox = gr.Textbox(label="Enter your question", value="What is Medicare advantage ?")
    with gr.Row():
        button = gr.Button("Submit", variant="Primary")
    with gr.Column():
        output1 = gr.Textbox(lines=1, max_lines=10, label="Raw Vector Search output")   
        output2 = gr.Textbox(lines=1, max_lines=10, label="LLM output")
    button.click(query_data, textbox, outputs=[output1, output2])
demo.launch()