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

### Q3. Describe the Neural Language Model.      

### Answer:      

A Neural Language Model (NLM) is a type of language model that utilizes neural networks to predict the likelihood of a sequence of words. These models represent a significant advancement over traditional statistical language models, such as n-gram models, primarily because they can capture longer dependencies and manage larger contexts effectively.     

#### Key Features of Neural Language Models:     

1. Word Embeddings:       
At the core of NLMs are word embeddings, which transform words into dense vector representations. These embeddings capture semantic meanings and are learned during the training process.      

2. Handling of Context:       
Unlike n-gram models that explicitly count words and their frequencies within a fixed window, NLMs learn to predict words based on a dynamically learned context, capturing more nuanced language patterns.      

3. Model Architectures:         

There are several types of neural network architectures used in NLMs:            

Recurrent Neural Networks (RNNs):       
Designed to handle sequences of data, RNNs process words one at a time while maintaining a 'memory' (hidden state) of what has been processed so far. This allows them to manage dependencies between words in a sentence.      

Long Short-Term Memory Networks (LSTMs) and Gated Recurrent Units (GRUs):       
These are specialized RNNs that have mechanisms to learn dependencies over longer distances, making them effective for language modeling.      

Transformers:       
A newer architecture that uses self-attention mechanisms to weigh the influence of different words, regardless of their position in the input sequence. This architecture has become the basis for models like BERT and GPT, significantly enhancing the capabilities of language models.      

#### Training Process:          

Neural Language Models are typically trained on large corpora of text. During training, the model is presented with sequences of words and learns to predict the next word in the sequence. This is done by adjusting the model weights to minimize the prediction error, often using backpropagation and optimization techniques like stochastic gradient descent.      

#### Applications:        

Neural Language Models are used in a wide range of NLP tasks, including:      

1. Text Generation:       
Generating coherent and contextually relevant text based on a prompt.
2. Speech Recognition:         
Transforming spoken language into text by predicting sequences of words.
3. Machine Translation:          
Translating text from one language to another.
4. Sentiment Analysis:           
Determining the sentiment expressed in a piece of text.        

Overall, Neural Language Models have revolutionized the field of NLP by providing tools that understand and generate human-like text, leveraging deep learning technologies to capture complex language patterns more effectively than ever before.

### Q4. What is a bias? In word embedding and how to do debiasing?       

### Answer:       

#### What is Bias in Word Embeddings?          

Bias in word embeddings refers to systematic, often prejudicial associations learned by the models from the training data. This data, typically large corpora of human-generated text, may reflect societal biases related to gender, race, ethnicity, or other factors. For example, traditional gender stereotypes might be seen in embeddings where words like "nurse" or "secretary" are more closely associated with female pronouns than male ones, while "engineer" or "pilot" might be closer to male pronouns.

These biases in word embeddings can lead to discriminatory or biased outcomes in downstream NLP applications, such as hiring tools, search engines, or chatbots, thereby perpetuating and even amplifying existing social biases.

#### How to Debias Word Embeddings        

Debiasing word embeddings involves methods to reduce or eliminate biased representations. Several approaches have been developed:

1. Bias Identification:         
The first step is identifying the bias vectors in the embeddings. One common technique involves isolating the direction corresponding to a specific bias (e.g., gender) in the vector space. This can be done by analyzing differences in vectors between stereotypically paired words (like "man" - "woman", "brother" - "sister").         

2. Neutralization and Equalization:         

Neutralization:           
This involves adjusting word vectors so that neutral words are made orthogonal to the bias direction. For example, words such as "nurse" or "engineer" are adjusted to be gender-neutral by removing their projections on the gender vector.     

Equalization:            
Ensuring that specific sets of words are equidistant to contrasting words. For instance, ensuring "grandmother" and "grandfather" are both equally distant to "family" and similar terms.          

3. Counterfactual Data Augmentation:            
Generating synthetic training data where terms with potential biases are swapped with their counterparts, such as changing "he is a nurse" to "she is a nurse" and vice versa in training datasets to help the model learn unbiased representations.         

4. Constraint-Based Training:           
Modifying the training objective of the embedding model to include constraints that minimize bias. This could involve adding regularization terms that penalize the model for learning biased representations.          

5. Post-Processing Adjustments:           
Applying modifications to pre-trained word embeddings before they are used in applications. This can involve techniques like the aforementioned neutralization and equalization, applied after the model has been trained.           

6. Bias-Aware Model Architecture:            
Designing NLP models and algorithms that are inherently less likely to amplify biases. This might include attention mechanisms that can learn to ignore biased aspects of the data.          

#### Importance of Debiasing          

Debiasing is crucial for building fair, ethical, and socially responsible AI systems. By actively addressing and reducing bias in word embeddings, developers can help ensure that AI technologies do not reinforce undesirable stereotypes and offer equal performance across diverse demographic groups. However, complete debiasing is challenging and an area of active research, emphasizing the need for continuous improvement and vigilance in deploying NLP models.

### Q5. How does modern machine translation work using the language model?      

### Answer:         

