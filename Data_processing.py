import os
import pickle
import numpy as np
import polars as pl
import pandas as pd
import random
np.random.seed(0)

def create_train_test_df(ratings):
    # The Data Structure
    grouped_users = ratings.group_by('userId', maintain_order=True).agg(pl.col('movieId'), pl.col('rating'))
    grouped_movies= ratings.group_by('movieId', maintain_order=True).agg(pl.col('userId') , pl.col('rating'))
    user_idx_map = {user_id: idx for idx, user_id in enumerate(grouped_users['userId'].to_list())}
    movie_idx_map = {movie_id: idx for idx, movie_id in enumerate(grouped_movies['movieId'].to_list())}
    del grouped_users , grouped_movies
 
    # Create inverse mapping for user indices
    idx_to_user = {idx: user_id for user_id, idx in user_idx_map.items()}

    # Create inverse mapping for movie indices
    idx_to_movie = {idx: movie_id for movie_id, idx in movie_idx_map.items()}

    # Add a column using a custom function
    ratings = ratings.with_columns([
        pl.col("userId").cast(pl.Int32),
        pl.col("movieId").cast(pl.Int32),
        pl.col("rating").cast(pl.Float32),
        pl.col("userId").replace_strict(user_idx_map).alias("userIdx"),
        pl.col("movieId").replace_strict(movie_idx_map).alias("movieIdx"),])
        # pl.col("userId").map_elements(indexing_users, return_dtype = pl.Int32).alias("userIdx"),
        # pl.col("movieId").map_elements(indexing_movies,return_dtype = pl.Int32).alias("movieIdx"),
    # method 2 - very good
    def train_test_split_df(df, seed=0, test_size=0.1):
        return df.with_columns(
            pl.int_range(pl.len(), dtype=pl.UInt32)
            .shuffle(seed=seed)
            .gt(pl.len() * test_size)
            .alias("split")
        ).partition_by("split", include_key=False)


    def train_test_split(X,seed=0, test_size=0.1):
        (X_train, X_test) = train_test_split_df(X, seed=seed, test_size=test_size)
        return (X_train, X_test)

    train_df,test_df = train_test_split(ratings)
    return  train_df,test_df,user_idx_map,  movie_idx_map ,idx_to_user, idx_to_movie

def save_idx_maps(output_data_folder,user_idx_map,  movie_idx_map ,idx_to_user,idx_to_movie,genre_to_idx,specific_indices):
    # Create the directory if it doesn't exist
    os.makedirs(output_data_folder, exist_ok=True)
    # Save the dictionaries
    with open(os.path.join(output_data_folder, 'user_idx_map.pkl'), 'wb') as f:
        pickle.dump(user_idx_map, f)

    with open(os.path.join(output_data_folder, 'movie_idx_map.pkl'), 'wb') as f:
        pickle.dump(movie_idx_map, f)

    with open(os.path.join(output_data_folder, 'idx_to_user.pkl'), 'wb') as f:
        pickle.dump(idx_to_user, f)

    with open(os.path.join(output_data_folder, 'idx_to_movie.pkl'), 'wb') as f:
        pickle.dump(idx_to_movie, f)

    with open(os.path.join(output_data_folder, 'genre_to_idx.pkl'), 'wb') as f:
        pickle.dump(genre_to_idx, f)

    with open(os.path.join(output_data_folder ,'specific_indices.pkl'), 'wb') as f:
        pickle.dump(specific_indices, f)
    print("Dictionaries saved successfully!")


def Save_data_split(experiment_folder,users_train,users_test,users_test_idxes,movies_test_idxes,users_train_idxes,movies_train_idxes,movies_train,movies_test,movies_genres_array):
    os.makedirs(experiment_folder, exist_ok=True)
    np.save(f'./{experiment_folder}/users_train.npy', users_train,allow_pickle=True)
    np.save(f'./{experiment_folder}/users_test.npy', users_test,allow_pickle=True)

    np.save(f'./{experiment_folder}/users_test_idxes.npy', users_test_idxes,allow_pickle=True)
    np.save(f'./{experiment_folder}/movies_test_idxes.npy', movies_test_idxes,allow_pickle=True)

    np.save(f'./{experiment_folder}/users_train_idxes.npy', users_train_idxes,allow_pickle=True)
    np.save(f'./{experiment_folder}/movies_train_idxes.npy', movies_train_idxes,allow_pickle=True)

    np.save(f'./{experiment_folder}/movies_train.npy', movies_train,allow_pickle=True)
    np.save(f'./{experiment_folder}/movies_test.npy', movies_test,allow_pickle=True)


    np.save(f'./{experiment_folder}/movies_genres_array.npy', movies_genres_array,allow_pickle=True)


    print("Train data saved successfully!")

def split(dataframe):

  grouped_users = dataframe.group_by('userIdx', maintain_order=True).agg(pl.col('movieIdx'), pl.col('rating'))
  users_data = grouped_users[:,1:].to_numpy()
  user_indices = grouped_users[:,0].to_numpy()

  grouped_movies= dataframe.group_by('movieIdx', maintain_order=True).agg(pl.col('userIdx') , pl.col('rating'))
  movies_data = grouped_movies[:,1:].to_numpy()
  movies_indices = grouped_movies[:,0].to_numpy()

  return users_data,user_indices,movies_data,movies_indices

def create_movie_data(movies_df,movie_idx_map):
    def parse_genre(values):
        all_genres = []
        for val in values:
            val_list = val.split('|')
            new = [x for x in val_list if x not in all_genres]
            all_genres = all_genres + new
        all_genres.pop(-1)
        return all_genres
    all_genres = parse_genre(movies_df['genres'].tolist())
    genre_to_idx = {genre: idx for idx, genre in enumerate(all_genres)}
    def parser(val):
        init = np.zeros(len(all_genres), dtype=int)
        val_list = val.split('|')
        if val_list[0] != '(no genres listed)':
            indices = [all_genres.index(v) for v in val_list]
            init[indices] = 1
        return init
    movies_df['feature_vector'] = movies_df['genres'].apply(parser)
    movieid_to_feature_vector = {a:b for (a, b) in zip(movies_df['movieId'], movies_df['feature_vector'])}
    feature_vectors = np.zeros((len(movieid_to_feature_vector), 19), dtype=int)

    for val in movieid_to_feature_vector.keys():
        try:
            movie_index = movie_idx_map[val]
            feature_vectors[movie_index] = movieid_to_feature_vector[val]
        except KeyError:
            continue

    return movies_df, feature_vectors,genre_to_idx,movieid_to_feature_vector ,all_genres

def get_specific_genre_idxs(movies_df, genre_list ,movieid_to_index  ,all_genres ):
    result = dict()
    for genre in genre_list:
        ids = []
        idxes = []
        for id, val in zip(movies_df['movieId'], movies_df['feature_vector']):
            if val[all_genres.index(genre)] == 1:
                ids.append(id)

        for id in ids:
            try:
                idxes.append(movieid_to_index[id])
            except KeyError:
                continue
        result[genre] = list(np.array(idxes,dtype =int))
    return result
     

if __name__ == "__main__":
    dataset = "ml-32m"
    source_data_folder = f"Data/{dataset}"
    ratings = pl.read_csv(f"{source_data_folder}/ratings.csv")
    movies_df = pd.read_csv(f"{source_data_folder}/movies.csv")
    output_data_folder=f"Training_data/{dataset}"


    train_df,test_df,user_idx_map,movie_idx_map ,idx_to_user, idx_to_movie = create_train_test_df(ratings)
    users_test,users_test_idxes,movies_test,movies_test_idxes = split(test_df)
    users_train,users_train_idxes,movies_train,movies_train_idxes = split(train_df)
    movies_df, movies_genres_array,genre_to_idx,movieid_to_feature_vector,all_genres = create_movie_data(movies_df,movie_idx_map)
    specific_indices = get_specific_genre_idxs(movies_df= movies_df , genre_list = all_genres ,movieid_to_index = movie_idx_map ,all_genres = all_genres )
    # print(type(specific_indices))
    Save_data_split(output_data_folder,users_train,users_test,users_test_idxes,movies_test_idxes,users_train_idxes,movies_train_idxes,movies_train,movies_test,movies_genres_array)
    save_idx_maps(output_data_folder,user_idx_map,  movie_idx_map ,idx_to_user,idx_to_movie,genre_to_idx,specific_indices)  