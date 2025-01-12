import numpy as np
import pandas as pd
import polars as pl
import time
from src.Latent_Factor_Updates import Update_movie_factors_with_features, Update_user_biases, Update_user_factors, Update_movie_biases, update_feature_factors, calc_metrics
from src.Helper_Functions import load_model, Load_idx_maps, create_fake_user, setup_logging, setup_experiment_folder, Load_training_data, Load_idx_maps, Load_test_data, get_possible_movie_indices

def create_idx_to_title(movies_csv_path, movie_idx_map):
    # Load the movies.csv file into a DataFrame
    movies_df = pd.read_csv(movies_csv_path)

    # Ensure columns 'movieId' and 'title' exist in the DataFrame
    if 'movieId' not in movies_df.columns or 'title' not in movies_df.columns:
        raise ValueError("The CSV file must contain 'movieId' and 'title' columns.")

    # Create a mapping from movie_id to title
    movie_id_to_title = dict(zip(movies_df['movieId'], movies_df['title']))

    # Create the idx_to_title dictionary using movie_idx_map
    idx_to_title = {idx: movie_id_to_title[movie_id] for movie_id, idx in movie_idx_map.items() if movie_id in movie_id_to_title}

    return idx_to_title

# Paths and data loading
movies_csv_path = "Data/ml-32m/movies.csv"  # Path to the movies.csv file
ratings = pd.read_csv("Data/ml-32m/ratings.csv")
source_data_folder = "Training_data/ml-32m"

user_idx_map, movie_idx_map, idx_to_user, idx_to_movie, genre_to_idx, specific_indices = Load_idx_maps(source_data_folder)
movies_factors, users_factors, user_bias, item_bias, feature_vectors = load_model("Experiments_ml-32m/B_U_V_F/")
idx_to_title = create_idx_to_title(movies_csv_path, movie_idx_map)

# Load training data
users_train, movies_train, movies_train_idxes, users_train_idxes, movies_genres_array = Load_training_data(source_data_folder)

# Calculate mean and standard deviation for movie ratings
movies_mean = []
movies_std = []
for rate in movies_train[:, 1]:
    movies_mean.append(float(np.mean(rate)))
    movies_std.append(float(np.std(rate)))

movies_mean = np.array(movies_mean)
movies_std = np.array(movies_std)

# Identify most and least polarizing movies by standard deviation
most_polarizing_movies_idx_std = np.argsort(movies_std)[-50:][::-1]  # Top 50 by descending std
least_polarizing_movies_idx_std = np.argsort(movies_std)[:50]        # Bottom 50 by ascending std

# Identify most and least polarizing movies by vector length
vector_lengths = np.linalg.norm(movies_factors, axis=1)
most_polarizing_movies_idx_length = np.argsort(vector_lengths)[-50:][::-1]  # Top 50 by descending length
least_polarizing_movies_idx_length = np.argsort(vector_lengths)[:50]        # Bottom 50 by ascending length

# Combine both criteria (intersection or union as needed)
def find_combined_polarizing_movies(std_indices, length_indices):
    return list(set(std_indices).intersection(length_indices))

def find_combined_non_polarizing_movies(std_indices, length_indices):
    return list(set(std_indices).intersection(length_indices))

most_combined_polarizing_idx = find_combined_polarizing_movies(most_polarizing_movies_idx_std,vector_lengths)
print(most_combined_polarizing_idx)