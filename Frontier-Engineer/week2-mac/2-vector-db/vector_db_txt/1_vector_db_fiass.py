import numpy as np
import faiss
from sentence_transformers import SentenceTransformer



"""

1_vector_db_fiass.py:
Demonstrates how to generate sentence embeddings and use FAISS for efficient similarity search over text data.  
Shows end‑to‑end flow: text → embeddings → vector index → nearest neighbour retrieval based on semantic similarity.


**Flow of the Code**

* First you get the model
** SentenceTransformer model is loaded
** It converts text into numbers (vectors)
** Example:
** "Apple is fruit" → [0.12, 0.45, 0.78 ...]

* Then you create a model client
** Texts are converted into embeddings
** Each sentence becomes a vector
** Example:
** "FAISS is search" → [0.11, 0.22, 0.33]
** "Vectors store meaning" → [0.10, 0.25, 0.30]
** Dimension = size of each vector (like 384 numbers)


** FAISS index acts like a search system
** Index stores all vectors
** Example:
** Store vectors of 5 sentences in FAISS
** Now FAISS can compare vectors and find similar ones

** Instead, query is converted to vector
** Example:
** Query: "What is FAISS?"
** → [0.12, 0.21, 0.31]

** FAISS compares this query vector with stored vectors
** It calculates distance (difference)
** Smaller distance = more similar meaning

** Example result:
** Rank 1 → "FAISS is a library for efficient similarity search."
** Rank 2 → "Vectors represent data in numerical form."

* Finally you close the client
** Not needed here
** Everything runs locally
** No external connection to close

**Simple Analogy**

* Text → Sentence  
* Embedding → Fingerprint of meaning  
* FAISS → Search engine for fingerprints  
* Query → Your question  
* Distance → How close meanings are  
* Top K → Best matching answers  

"""

try:
    # Load a pre-trained model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("SentenceTransformer model loaded.")

    # Sample custom data (text)
    texts = [
        "FAISS is a library for efficient similarity search.",
        "Vectors represent data in numerical form.",
        "Embedding models convert text to vectors.",
        "Local vector databases can be faster for small datasets.",
        "FAISS supports both CPU and GPU operations."
    ]

    # Convert texts to vectors
    embeddings = model.encode(texts)
    print(f"Converted {len(texts)} texts to embeddings.")

    # Define vector dimension
    # Each embedding = list of numbers. dimension = how many numbers are in one vector
    # [0.1, 0.5, ...,] is 384 numbers and dimension = 384
    dimension = embeddings.shape[1]
    print(f"Vector dimension: {dimension}")

    # Create a FAISS index
    # This line creates a vector search index that finds similar vectors using distance
    index = faiss.IndexFlatL2(dimension)

    index.add(embeddings)

    print(f"Created FAISS index and added {len(embeddings)} vectors.")

    # Example query
    query_text = "What is FAISS?"
    query_vector = model.encode([query_text])[0]
    print(f"Encoded query: '{query_text}'")

    # Perform the query
    k = 3  # number of nearest neighbours to retrieve
    distances, indices = index.search(np.array([query_vector]), k)

    # Process and print results
    print("\nQuery Results:")
    for i in range(len(indices[0])):
        print(f"Rank: {i+1}")
        print(f"Text: {texts[indices[0][i]]}")
        print(f"Distance: {distances[0][i]}")
        print()

except Exception as e:
    print(f"An error occurred: {e}")