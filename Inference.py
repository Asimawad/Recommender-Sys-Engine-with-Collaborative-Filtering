import numpy as np
import pandas as pd
import pickle
from src.Helper_Functions import load_model, Load_idx_maps,create_fake_user

# Define constants
MOVIE_FILE_PATH = 'Data/ml-32m/movies.csv'

K_FACTORS = 32
LAMBDA_REG = 0.5
GAMMA = 0.5
TAW = 10

experiment_folder = "Experiments_ml-32m/B_U_V_F/"
data_folder = 'Training_data/ml-32m'
# K_factors = 30; lambda_reg = 1 ; gamma = 0.01 ; taw =  10


# Function to load data
def load_data():
    movies = pd.read_csv(MOVIE_FILE_PATH)
    movies_factors,users_factors,user_bias,item_bias,feature_vectors = load_model(experiment_folder)
    user_idx_map, movie_idx_map, idx_to_user, idx_to_movie, genre_to_idx, specific_indices = Load_idx_maps(data_folder)
    return movies, movies_factors, item_bias, movie_idx_map, idx_to_movie

# Function to get movie details
def get_movie_details(movies, title):
    filtered_movie = movies[movies['title'] == title]
    if not filtered_movie.empty:
        movie_name = filtered_movie['title'].values[0]
        movie_id = filtered_movie['movieId'].values[0]
        genre = filtered_movie['genres'].values[0]
        return movie_name, movie_id, genre
    else:
        return None, None, None

# Function to create a fake user
def create_user(movie_idx_map, movieId, item_bias, movies_factors):
    movie_idx = movie_idx_map[movieId.item()]
    list_of_favourite_movies = [movie_idx, 5]
    user = create_fake_user(list_of_favourite_movies, item_bias, movies_factors, K_FACTORS, LAMBDA_REG, GAMMA, TAW    )
    return user

# Function to generate movie recommendations
def generate_recommendations(dummy_user_vector, movies_factors, item_bias, idx_to_movie, movies, movie_name):
    recommendation_scores = (movies_factors @ dummy_user_vector) + (item_bias * 0.05)
    top_movie_indices = np.argsort(recommendation_scores)[::-1][:50]
    top_movie_ids = [idx_to_movie[idx] for idx in top_movie_indices]

    recommendations = {'title': [], 'genre': []}
    for MID in top_movie_ids:
        filtered_movie = movies[movies['movieId'] == MID]
        if not filtered_movie.empty:
            movieId, title, genre = (
                filtered_movie['movieId'].values[0],
                filtered_movie['title'].values[0],
                filtered_movie['genres'].values[0]
            )
            if title != movie_name:
                recommendations['title'].append(title)
                recommendations['genre'].append(genre)
    return pd.DataFrame(recommendations) , top_movie_indices

# Main function
def main():
    # Load data
    movies, movies_factors, item_bias, movie_idx_map, idx_to_movie = load_data()

    # Get movie details  : Uncomment any movie
    # movie_name, movie_id, genre = get_movie_details(movies, "Harry Potter and the Sorcerer's Stone (a.k.a. Harry Potter and the Philosopher's Stone) (2001)")
    # movie_name, movie_id, genre = get_movie_details(movies, "Lord of the Rings, The (1978)")
    # movie_name, movie_id, genre = get_movie_details(movies, "Avengers: Infinity War - Part II (2019)")
    # movie_name, movie_id, genre = get_movie_details(movies, "Saw (2004)")# "Casino (1995)"
    # movie_name, movie_id, genre = get_movie_details(movies, "Casino (1995)")
    # movie_name, movie_id, genre = get_movie_details(movies, "Twilight Saga: Breaking Dawn - Part 2, The (2012)")
    # movie_name, movie_id, genre = get_movie_details(movies, "Pirates of the Caribbean: Dead Man's Chest (2006)")
    movie_name, movie_id, genre = get_movie_details(movies, "Scooby-Doo (2002)")
    
    # movie_name, movie_id, genre = get_movie_details(movies, "Shrek 2 (2004)")

    
    if movie_name is None:
        print("Movie not found.")
        return

    print(f"The user liked the movie: {movie_name}, with id: {movie_id}, The genre is {genre}")

    # Create a fake user
    dummy_user_vector = create_user(movie_idx_map, movie_id, item_bias, movies_factors)

    # Generate recommendations
    top_movies , top_movie_indices = generate_recommendations(dummy_user_vector, movies_factors, item_bias, idx_to_movie, movies, movie_name)

    # Print top recommendations
    print("Recommendations are:\n")
    print(top_movies.head(25))
    with open(f'Archive/{movie_name}_Suggestions.pkl', 'wb') as f:
        pickle.dump(top_movie_indices, f)
# Run the main function
if __name__ == "__main__":
    main()