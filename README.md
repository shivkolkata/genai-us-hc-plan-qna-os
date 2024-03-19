# genai-us-hc-plan-qna-os

# Installation Steps

1. Install Qdrant Vector DB in local docker environment, below are the command - 

1.a. Docker Pull - 
docker pull qdrant/qdrant

1.b. Docker Run - 
docker run -p 6333:6333 -p 6334:6334 \                                                 
    -v $(pwd)/qdrant_storage:/qdrant/storage:z \
    qdrant/qdrant
           _                 _    
  __ _  __| |_ __ __ _ _ __ | |_  
 / _` |/ _` | '__/ _` | '_ \| __| 
| (_| | (_| | | | (_| | | | | |_  
 \__, |\__,_|_|  \__,_|_| |_|\__| 
    |_|                           

Version: 1.8.1, build: 3fbe1cae

1.c Access web UI at http://localhost:6333/dashboard

2. Create python virtual environment - conda create -p venv python==3.9 -y

2. Install packages - pip install -r requirements.txt

3. Enter the following parameters in .env file (.env file need to be created)
qdrant_url=http://localhost:6333
hf_token="<your own hugging face token>"
embedding_model_name="sentence-transformers/all-MiniLM-L6-v2" (you may use your own embedding model)
llm_model_name="mistralai/Mistral-7B-Instruct-v0.2" (you may use your own llm model)

4. Load Data, Convert to embeddings and insert into Mongo DB - python load_data.py

5. Navigate to Qdrant UI (http://localhost:6333/dashboard), following collection should get created - collection_of_text_blobs_with_chunks

6. For testing - python extract_information.py

7. Open browser navigate to - http://127.0.0.1:7860 for entering question (sample question - "What is Medicare advantage ?")
