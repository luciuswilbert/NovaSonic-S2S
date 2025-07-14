import os
import json
import time
import faiss
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv
import boto3

# Load environment variables
load_dotenv()

AWS_KEY_ID = os.getenv('AWS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
AWS_REGION = os.getenv('AWS_REGION')
AWS_EMBEDDING_REGION = os.getenv('AWS_EMBEDDING_REGION', AWS_REGION)
AWS_EMBEDDING_MODELID = os.getenv('AWS_EMBEDDING_MODELID')

FAISS_INDEX_FILE = 'faiss_index.bin'
FAISS_MAPPING_FILE = 'faiss_mapping.json'
CHUNKS_FILE = os.path.join('extractedPDFs', 'faiss_ready_data.json')

# Helper: Call Bedrock embedding model

def get_bedrock_embedding(text, client, model_id):
    """
    Call AWS Bedrock embedding model and return the embedding vector.
    """
    # This assumes Bedrock Titan Embeddings v1 API structure
    response = client.invoke_model(
        modelId=model_id,
        body=json.dumps({"inputText": text}),
        accept="application/json",
        contentType="application/json"
    )
    result = json.loads(response['body'].read())
    # Adjust this if your model returns a different structure
    return result['embedding']

# Main embedding and FAISS build

def main():
    # Load chunks
    with open(CHUNKS_FILE, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    print(f"Loaded {len(chunks)} chunks from {CHUNKS_FILE}")

    # Setup Bedrock client
    session = boto3.Session(
        aws_access_key_id=AWS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_EMBEDDING_REGION
    )
    bedrock = session.client('bedrock-runtime', region_name=AWS_EMBEDDING_REGION)

    # Get embedding dimension (first call)
    print("Getting embedding dimension...")
    first_vec = get_bedrock_embedding(chunks[0]['text'], bedrock, AWS_EMBEDDING_MODELID)
    embedding_dim = len(first_vec)
    print(f"Embedding dimension: {embedding_dim}")

    # Prepare arrays
    embeddings = np.zeros((len(chunks), embedding_dim), dtype='float32')
    id_to_metadata = {}

    # Embed all chunks
    for i, chunk in enumerate(tqdm(chunks, desc="Embedding chunks")):
        try:
            vec = get_bedrock_embedding(chunk['text'], bedrock, AWS_EMBEDDING_MODELID)
            embeddings[i] = np.array(vec, dtype='float32')
            id_to_metadata[chunk['id']] = chunk['metadata']
        except Exception as e:
            print(f"Error embedding chunk {chunk['id']}: {e}")
            embeddings[i] = np.zeros(embedding_dim, dtype='float32')
        time.sleep(0.1)  # To avoid throttling; adjust as needed

    # Build FAISS index
    print("Building FAISS index...")
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(embeddings)
    faiss.write_index(index, FAISS_INDEX_FILE)
    print(f"FAISS index saved to {FAISS_INDEX_FILE}")

    # Save mapping
    with open(FAISS_MAPPING_FILE, 'w', encoding='utf-8') as f:
        json.dump(id_to_metadata, f, indent=2, ensure_ascii=False)
    print(f"Chunk ID to metadata mapping saved to {FAISS_MAPPING_FILE}")

if __name__ == "__main__":
    main() 