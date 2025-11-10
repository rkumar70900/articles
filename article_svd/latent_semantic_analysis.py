import numpy as np
from numpy.linalg import svd
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import matplotlib.pyplot as plt

terms = ["car", "automobile", "truck", "vehicle", "dog", "cat"]

# Fake document-term matrix (counts)
docs = np.array([
    [5, 4, 0, 5, 0, 0],  # auto themed
    [4, 5, 1, 4, 0, 0],
    [0, 0, 5, 0, 0, 0],  # truck doc
    [0, 0, 1, 0, 4, 4],  # dog/cat doc
])

U, S, Vt = svd(docs, full_matrices=False)
k = 2  # latent dimensions
term_vectors = (Vt[:k].T * S[:k])

sim = cosine_similarity(term_vectors)

# Heatmap for word similarity
plt.figure(figsize=(6,5))
sns.heatmap(sim, annot=True, xticklabels=terms, yticklabels=terms)
plt.title("LSA Word Similarity (SVD)")
plt.show()