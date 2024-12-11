import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

from collections import Counter
data = []
with open("data/articles.json") as file:
    json_data = json.load(file)
    for item in json_data:
        data.append((item["source"], item["article"]))

if data:




    #Plot heatmap using seaboct the second value (strings) from the tuples
    texts = [item[1] for item in data]

    # Convert texts to TF-IDF vectors
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)

    # Compute cosine similarity matrix
    similarity_matrix = cosine_similarity(tfidf_matrix)
    # Convert the similarity matrix to a DataFrame for better visualization
    df_similarity = pd.DataFrame(similarity_matrix, index=[item[0] for item in data], columns=[item[0] for item in data])

    # Create a heatmap
    plt.figure(figsize=(8, 6))

    sns.heatmap(df_similarity, annot=True, cmap='coolwarm', xticklabels=True, yticklabels=True)
    plt.title("News Similarity")
    plt.savefig("heatmap.png")

