import numpy as np
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from sentence_transformers import SentenceTransformer
from PIL import Image
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from networkx import karate_club_graph, spring_layout
from node2vec import Node2Vec
import librosa

"""
2_different_types_of_embeddings.py:
Demonstrates how to generate embeddings at two levels—Word2Vec for individual words and SentenceTransformer for full sentences.  
Shows how raw text (words/sentences) is converted into numerical vectors that capture semantic meaning.


**Flow of the Code**

* First you load the Word2Vec model
** Training data: [["cat", "say", "meow"], ["dog", "say", "woof"]]
** Word2Vec learns relationships between words
** vector_size=10 → each word becomes 10 numbers
** window=5 → looks at 5 nearby words to learn context
** min_count=1 → includes all words even if they appear once
** Example:
** "cat" → [0.2, 0.8, 0.1, ... 10 numbers]
** "meow" → [0.22, 0.79, 0.12, ... 10 numbers]
** cat and meow vectors are close → because they appear together

* Then you print word embeddings
** Loop through each word: cat, dog, say, meow, woof
** model.wv[word] → returns the vector for that word
** Example:
** model.wv["cat"] → [0.2, 0.8, 0.1 ...]
** model.wv["dog"] → [0.25, 0.75, 0.15 ...]

* Then you load the SentenceTransformer model
** Model name: "paraphrase-MiniLM-L6-v2"
** This model understands full sentences (not just words)
** It is pre-trained on millions of sentences
** Example:
** Word2Vec → "cat" → one vector
** SentenceTransformer → "cat is cute" → one vector for full sentence

* Then you encode sentences
** Two sentences are converted to vectors
** model.encode(sentences) → returns array of embeddings
** embeddings.shape → shows size like (2, 384)
** First 5 values of first sentence are printed
** Example:
** "This is an example sentence" → [0.03, -0.12, 0.45, 0.08, -0.22 ...]

* Finally you run the main block
** if __name__ == "__main__" starts execution
** First calls word_embeddings()
** Then calls sentence_embeddings()
** Output is printed to console

**Simple Analogy**

* Word2Vec → Dictionary that stores meaning as numbers
* SentenceTransformer → Translator that converts full sentences to numbers
* vector_size → Length of the secret code
* window → How far the model looks around a word
* encode() → Convert text to numbers
* Similar vectors → Similar meaning

"""

# Word Embeddings
def word_embeddings():
    # Word2Vec is a popular method for generating word embeddings
    # It learns vector representations of words that capture semantic relationships
    sentences = [['cat', 'say', 'meow'], ['dog', 'say', 'woof']]
    model = Word2Vec(sentences, vector_size=10, window=5, min_count=1, workers=4)
    # Parameters:
    # - vector_size=10: Dimensionality of the word vectors
    # - window=5: Maximum distance between current and predicted word within a sentence
    # - min_count=1: Ignores all words with total frequency lower than this
    # - workers=4: Number of CPU cores to use for training
    
    for word in ['cat', 'dog', 'say', 'meow', 'woof']:
        print(f"Word Embedding for '{word}':", model.wv[word])

# Sentence Embeddings
def sentence_embeddings():
    # SentenceTransformer is a library for state-of-the-art sentence embeddings
    # It's based on BERT architecture and fine-tuned for generating sentence embeddings
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    # 'paraphrase-MiniLM-L6-v2' is the name of the pre-trained model being used
    sentences = ["This is an example sentence", "Each sentence is converted to a vector"]
    embeddings = model.encode(sentences)
    print("Sentence Embedding shape:", embeddings.shape)
    print("First sentence embedding:", embeddings[0][:5])  # First 5 dimensions


if __name__ == "__main__":
    print("Word Embeddings:")
    word_embeddings()
    
    print("\nSentence Embeddings:")
    sentence_embeddings()
    