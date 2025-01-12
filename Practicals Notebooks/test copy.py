import os
import pickle

def Load_training_data(source_data_folder):
    # List of filenames corresponding to training data
    file_names = [
        'users_train.pkl',
        'movies_train.pkl',
        'movies_train_idxes.pkl',
        'users_train_idxes.pkl',
        'movies_genres_array.pkl'
    ]
    
    # Load each file and return as a tuple
    training_data = tuple(
        pickle.load(open(os.path.join(source_data_folder, file_name), 'rb'))
        for file_name in file_names
    )
    return training_data

def Load_test_data(source_data_folder):
    # List of filenames corresponding to test data
    file_names = [
        'users_test.pkl',
        'movies_test.pkl',
        'users_test_idxes.pkl',
        'movies_test_idxes.pkl'
    ]

    # Load each file and return as a tuple
    test_data = tuple(
        pickle.load(open(os.path.join(source_data_folder, file_name), 'rb'))
        for file_name in file_names
    )
    return test_data
source_data_folder = 'Training_data/ml-32m'
users_train, movies_train, movies_train_idxes, users_train_idxes, movies_genres_array = Load_training_data(source_data_folder)
users_test, movies_test, users_test_idxes, movies_test_idxes = Load_test_data(source_data_folder)
