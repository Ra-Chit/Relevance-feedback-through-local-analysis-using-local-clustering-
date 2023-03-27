import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
print("module imported")
docs = [
    "Natural language processing (NLP) is a field of computer science, artificial intelligence, and linguistics concerned with the interactions between computers and human languages.",
    "One of the key challenges in NLP is to develop algorithms that can understand and interpret the meaning of human language.",
    "NLP is used in a wide range of applications, including language translation, chatbots, and voice assistants.",
    "Some of the key techniques used in NLP include parsing, part-of-speech tagging, named entity recognition, and sentiment analysis.",
    "NLP has made significant advances in recent years, thanks to the development of deep learning techniques and the availability of large datasets.",
    "A key challenge in NLP is to deal with the ambiguity and variability of human language, which can make it difficult to accurately interpret meaning.",
    "NLP can be used to analyze and process both written and spoken language, making it a powerful tool for applications such as speech recognition and text analysis.",
    "Some of the key applications of NLP include automated translation, sentiment analysis, and question-answering systems.",
    "NLP can be used in a wide range of industries, including healthcare, finance, and e-commerce, to improve communication and enhance user experiences.",
    "As NLP continues to advance, it is likely to have a significant impact on a wide range of industries, from healthcare and finance to education and entertainment."
]
query = "nlp in preprocessing"

# Create TfidfVectorizer instance
vectorizer = TfidfVectorizer(stop_words='english')
# Vectorize docs and query
X = vectorizer.fit_transform(docs)
q = vectorizer.transform([query])

print(q)
# Cluster documents
k = 2
model = KMeans(n_clusters=k, init='k-means++', max_iter=100, n_init=1)
model.fit(X)

# Get cluster labels
labels = model.labels_

# Get documents in each cluster
cluster_docs = [[] for i in range(k)]
for i, label in enumerate(labels):
    cluster_docs[label].append(docs[i])

# Calculate cluster centroids
centroids = model.cluster_centers_
print(centroids)
cluster_scores = [(i, cluster_similarities[i] * correlation_factors[i]) for i in range(k)]
sorted_clusters = sorted(cluster_scores, key=lambda x: -x[1])

# Print expanded query
expanded_query = query
for i in range(k):
    if len(cluster_docs[sorted_clusters[i][0]]) > 0:
        expanded_query += ' ' + ' '.join(cluster_docs[sorted_clusters[i][0]])
print(expanded_query)
