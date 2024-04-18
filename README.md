# HW to Chapter 21 “Language Model”      

## Non-programming Assignment      

### Q1. Describe word embedding.     

### Answer      

Word embedding is a technique in natural language processing (NLP) that involves mapping words or phrases from a vocabulary to vectors of real numbers in a low-dimensional space relative to the vocabulary size. These vectors are intended to represent the meanings of the words in such a way that the geometric relationships between vectors correspond to semantic relationships between words. For example, words with similar meanings tend to have vectors that are close to each other in the embedding space, and relationships like analogies can often be represented as vector offsets.      

The process of creating word embeddings generally involves training a model on a large corpus of text. During this training, the model learns to associate words with contexts. Two popular models for creating word embeddings are:      

Word2Vec:       
Developed by Google, it offers two architectures: Continuous Bag of Words (CBOW) and Skip-Gram. CBOW predicts a target word from a set of context words, while Skip-Gram does the opposite, predicting context words from a target word.      

GloVe (Global Vectors for Word Representation):       
Developed by Stanford, it is an unsupervised learning algorithm for obtaining vector representations by aggregating global word-word co-occurrence statistics from a corpus.      

These embeddings can then be used as features in many NLP applications, such as sentiment analysis, machine translation, and information retrieval, allowing these applications to capture the semantic properties of words.      

### Q2. What is the measure of word similarity?      

### Answer:        

The measure of word similarity quantifies how closely related or similar two words are in terms of their meanings. This measurement is generally based on the distance between their vector representations in a word embedding space. There are several methods to calculate this similarity:

Cosine Similarity: This is the most commonly used metric in word embeddings. Cosine similarity measures the cosine of the angle between two vectors. The result ranges from -1 (meaning exactly opposite) to 1 (exactly the same), with higher values indicating greater similarity. A cosine similarity close to 1 suggests that the two words are very similar in the context of their use in the language.      

Euclidean Distance: Another approach is to use the Euclidean distance between two vectors. Unlike cosine similarity, a smaller Euclidean distance indicates greater similarity. This measure considers the magnitude of vectors in addition to their direction.      

Manhattan Distance: Also known as the L1 norm, this metric sums the absolute differences between the coordinates of the vectors. Like Euclidean distance, a smaller Manhattan distance indicates greater similarity.      

Jaccard Similarity: Although less common in the context of word embeddings, Jaccard similarity measures similarity as the size of the intersection divided by the size of the union of the sample sets.      

Among these, cosine similarity is preferred for word embeddings because it effectively captures the orientation of vectors and is less affected by the magnitude of the vectors. This makes it particularly suitable for comparing words based on their directional closeness in the embedding space, reflecting their semantic similarity.       

### Describe the Neural Language Model.      

### Answer:      

