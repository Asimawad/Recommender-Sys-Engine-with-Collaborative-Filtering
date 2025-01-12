import numpy as np
import pandas as pd
import pickle
from src.Helper_Functions import load_model, Load_idx_maps,create_fake_user

# Define constants 

K_FACTORS = 32
LAMBDA_REG = 2
GAMMA = 1
TAW = 100
dataset = "ml-32m"
model = "B_U_V_F"
MOVIE_FILE_PATH = f'Data/{dataset}/movies.csv'
experiment_folder = f"Experiments_{dataset}/{model}/"
data_folder = f'Training_data/{dataset}'

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
def main(query):
    # Load data
    movies, movies_factors, item_bias, movie_idx_map, idx_to_movie = load_data()
    
    # Get movie details  : Uncomment any movie
    movie_name, movie_id, genre = get_movie_details(movies, query)

    
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
    print(top_movies.head(10))
    with open(f'Archive/{movie_name}_Suggestions.pkl', 'wb') as f:
        pickle.dump(top_movie_indices, f)
# Run the main function
if __name__ == "__main__":
   
    moveis_query = ["Up (2009)","Harry Potter and the Sorcerer's Stone (a.k.a. Harry Potter and the Philosopher's Stone) (2001)","Toy Story (1995)","Pirates of the Caribbean: Dead Man's Chest (2006)","Twilight Saga: Breaking Dawn - Part 1, The (2011)","Casino (1995)","Saw VI (2009)","Avengers: Infinity War - Part I (2018)","Lord of the Rings, The (1978)"]
    for query in moveis_query:
        main(query=query)