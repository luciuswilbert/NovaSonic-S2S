import faiss
import numpy as np
import json
import os

# Adjust these if your files are in a different location
FAISS_INDEX_FILE = os.path.join('faiss_index', 'faiss_index.bin')
FAISS_MAPPING_FILE = os.path.join('faiss_index', 'faiss_mapping.json')

# Load FAISS index
index = faiss.read_index(FAISS_INDEX_FILE)
print(f"FAISS index loaded from {FAISS_INDEX_FILE}")

# Get number of vectors and dimension
num_vectors = index.ntotal
dimension = index.d
print(f"Number of vectors: {num_vectors}")
print(f"Vector dimension: {dimension}")

# Print a sample vector (first one)
if num_vectors > 0:
    sample_vector = np.zeros((dimension,), dtype='float32')
    index.reconstruct(0, sample_vector)
    print(f"Sample vector (first 10 dims): {sample_vector[:10]}")
else:
    print("No vectors found in the index.")

# Print a sample metadata if mapping file exists
if os.path.exists(FAISS_MAPPING_FILE):
    with open(FAISS_MAPPING_FILE, 'r', encoding='utf-8') as f:
        mapping = json.load(f)
    if mapping:
        sample_id = list(mapping.keys())[0]
        print(f"Sample chunk ID: {sample_id}")
        print(f"Sample metadata: {mapping[sample_id]}")
    else:
        print("Mapping file is empty.")
else:
    print(f"Mapping file {FAISS_MAPPING_FILE} not found.") 