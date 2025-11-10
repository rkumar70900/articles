import numpy as np
from numpy.linalg import svd
from sklearn.metrics.pairwise import cosine_similarity

# creating a matrix with 50 users and 40 movies
users, movies = 50, 40
# filling out a sparse matrix with random ratings
ratings = np.random.choice([0, 0, 0, 1, 2, 3, 4, 5], size=(users, movies))
# breaking down the ratings matrix into latent factors
U_r, S_r, Vt_r = svd(ratings, full_matrices=False)
# we need only the top 5 as recommendations
k_r = 5
user_latent = U_r[:, :k_r] * S_r[:k_r]
movie_latent = Vt_r[:k_r, :].T

print("\nRatings matrix shape:", ratings.shape)
print("User latent factors shape:", user_latent.shape)
print("Movie latent factors shape:", movie_latent.shape)