import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import svd

# Load your dataset
df = pd.read_csv("./movies_dataset.csv")  # replace filename

X = df.select_dtypes(include=[np.number]).to_numpy()
X = X - X.mean(axis=0)   # center

U, S, Vt = svd(X, full_matrices=False)

k = 7  # components you want
X_reduced = X @ Vt[:k].T

# Explained variance
explained = (S[:k]**2) / np.sum(S**2)

plt.plot(np.cumsum(explained))
plt.xlabel("Principal Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA via SVD")
plt.grid()
plt.show()

print("Original shape:", X.shape)
print("Reduced shape:", X_reduced.shape)
