import os
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

folder_path = r"C:\Users\Acer\Desktop\Document\documents"
docs = []
filenames = []

for file in os.listdir(folder_path):
    if file.endswith(".txt"):
        with open(os.path.join(folder_path, file), "r", encoding="utf-8") as f:
            docs.append(f.read())
            filenames.append(file)



vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(docs)



similarity_matrix = cosine_similarity(tfidf_matrix)

# Convert to DataFrame for readability
sim_df = pd.DataFrame(similarity_matrix, index=filenames, columns=filenames)

# Save CSV
sim_df.to_csv("similarity_matrix.csv")
print("Similarity Matrix Saved → similarity_matrix.csv")



plt.figure(figsize=(7,6))
plt.imshow(similarity_matrix, interpolation='nearest')
plt.xticks(ticks=np.arange(len(filenames)), labels=filenames, rotation=45, ha="right")
plt.yticks(ticks=np.arange(len(filenames)), labels=filenames)
plt.title("Cosine Similarity Heatmap")
plt.colorbar()
plt.tight_layout()

plt.savefig("similarity_heatmap.png")
print("Heatmap Saved → similarity_heatmap.png")


# 5. TF-IDF FEATURE TABLE SAVE

tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), index=filenames, columns=vectorizer.get_feature_names_out())
tfidf_df.to_csv("tfidf_features.csv")
print("TF-IDF Dataset Saved → tfidf_features.csv")

print("\nDONE! All outputs generated successfully.")
