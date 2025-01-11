# Movie Recommendation System using Collaborative Filtering

## Introduction

This project implements a movie recommendation system using collaborative filtering techniques, specifically matrix factorization. The system is designed to predict user preferences for movies based on historical ratings and to generate personalized movie recommendations.

By leveraging latent factor models, the system captures underlying patterns in user behavior and movie characteristics. The project uses the MovieLens dataset, which provides a rich source of user ratings and movie metadata.

---
 
## Table of Contents

- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Methodology](#methodology)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Usage](#usage)
  - [Generating Recommendations](#generating-recommendations)
  - [Visualizing Embeddings](#visualizing-embeddings)
- [Results](#results)
- [Visualization](#visualization)
- [References](#references)
- [Acknowledgments](#acknowledgments)

---

## Project Overview

The goal of this project is to build a scalable and efficient movie recommendation system using collaborative filtering. The key features include:

- **Data Preprocessing:** Handling large datasets (e.g., 32 million ratings) efficiently.
- **Model Training:** Implementing matrix factorization with bias terms and regularization.
- **Performance Optimization:** Utilizing techniques like Numba for just-in-time compilation to speed up computations.
- **Evaluation:** Tracking training and validation metrics, including loss and RMSE.
- **Recommendation Generation:** Providing personalized movie recommendations based on user input.
- **Visualization:** Visualizing movie embeddings to understand the latent feature space.

---

## Project Structure

The project is organized into the following main components:

- **Scripts:**
  - `main.py`: The main script that runs the training loop for the recommendation system.
  - `Data_Preprocessing.py`: Preprocesses the dataset and prepares training and test data.
  - `Recommendation.py`: Generates movie recommendations for a user based on their favorite movies.
  - `Visualization.py`: Visualizes movie embeddings using dimensionality reduction techniques.
- **Modules:**
  - `Factors_Update_Functions.py`: Contains functions for updating user and movie biases and latent factors.
  - `Helper_Functions.py`: Utility functions for plotting, logging, saving models, and loading data.
- **Data Directories:**
  - `ml-32m/`: Contains the original MovieLens dataset files (`ratings.csv`, `movies.csv`).
  - `TRAIN_TEST_DATA/`: Stores preprocessed training and test data.
- **Experiments Directory:**
  - `Experiments/`: Contains experiment logs, models, results, and plots.
- **Miscellaneous:**
  - `README.md`: This detailed readme file.
  - `requirements.txt`: Lists required Python packages.

---

## Installation

### Prerequisites

Ensure you have the following installed:

- Python 3.7 or higher
- [NumPy](https://numpy.org/)
- [Pandas](https://pandas.pydata.org/)
- [Numba](http://numba.pydata.org/)
- [Matplotlib](https://matplotlib.org/)
- [scikit-learn](https://scikit-learn.org/)
- [Polars](https://www.pola.rs/)
- [Pickle](https://docs.python.org/3/library/pickle.html)

### Steps

1. **Clone the Repository**

   ```bash
   git clone https://github.com/your_username/your_repository.git
   cd your_repository
   ```

2. **Set Up a Virtual Environment (Recommended)**

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install Required Packages**

   Install the required packages using `pip`:

   ```bash
   pip install -r requirements.txt
   ```

   If `requirements.txt` is not available, install packages individually:

   ```bash
   pip install numpy pandas numba matplotlib scikit-learn polars
   ```

---

## Data Preparation

### Dataset

The project uses the [MovieLens 25M Dataset](https://grouplens.org/datasets/movielens/25m/), which contains millions of user ratings for movies along with movie metadata.

**Required Files:**

- `ratings.csv`: Contains user ratings (`userId`, `movieId`, `rating`, `timestamp`).
- `movies.csv`: Contains movie metadata (`movieId`, `title`, `genres`).

### Steps to Prepare Data

1. **Download the Dataset**

   Download `ratings.csv` and `movies.csv` from the MovieLens website.

2. **Place the Files**

   Place the downloaded files in the `ml-32m/` directory within the project folder.

3. **Run Data Preprocessing**

   Execute the data preprocessing script:

   ```bash
   python Data_Preprocessing.py
   ```

   This script performs the following tasks:

   - Reads and processes the ratings and movies data.
   - Maps user and movie IDs to continuous indices starting from 0 for efficient storage and computation.
   - Splits the data into training and test sets (e.g., 80% training, 20% test).
   - Groups ratings by users and movies.
   - Saves the preprocessed data and mappings to the `TRAIN_TEST_DATA/` directory.

**Note:** The preprocessing step may take some time due to the size of the dataset.

---

## Methodology

### Collaborative Filtering and Matrix Factorization

The recommendation system uses **collaborative filtering**, which makes predictions about a user's interests by collecting preferences from many users. Specifically, it employs **matrix factorization** to discover latent features underlying the interactions between users and items (movies).

### Latent Factor Model

- **User Factors (`users_factors`):** Represents the latent preferences of users.
- **Movie Factors (`movies_factors`):** Represents the latent characteristics of movies.
- **Genre Factors (`genre_factors`):** Captures genre-specific characteristics.

### Bias Terms

- **User Bias (`user_bias`):** Captures the tendency of a user to rate items higher or lower than average.
- **Item Bias (`item_bias`):** Captures the inherent popularity or quality of an item.

### Objective Function

The model minimizes the regularized squared error between predicted and actual ratings:

\[
\min_{U, V, b_u, b_i} \sum_{(u, i) \in \text{Ratings}} \left( r_{ui} - (\mathbf{u}_u^\top \mathbf{v}_i + b_u + b_i) \right)^2 + \lambda \left( \|\mathbf{u}_u\|^2 + \|\mathbf{v}_i\|^2 \right) + \gamma \left( b_u^2 + b_i^2 \right)
\]

Where:

- \( r_{ui} \): Actual rating of user \( u \) for item \( i \).
- \( \mathbf{u}_u \): Latent factor vector for user \( u \).
- \( \mathbf{v}_i \): Latent factor vector for item \( i \).
- \( b_u \), \( b_i \): User and item biases.
- \( \lambda \), \( \gamma \): Regularization parameters.

### Optimization Technique

- **Alternating Least Squares (ALS):** The optimization alternates between fixing user factors and updating item factors, and vice versa.
- **Regularization:** Prevents overfitting by penalizing large latent factors and biases.

### Acceleration with Numba

- **Numba JIT Compilation:** The `@jit(nopython=True)` decorator is used to compile functions to machine code at runtime, significantly speeding up computations.

---

## Model Training

### Running the Training Script

Execute the training script:

```bash
python main.py
```

### Configurable Parameters

Within `main.py`, you can adjust the following hyperparameters:

- **Experiment Settings:**

  ```python
  experiment_name = "your_experiment_name"
  ```

- **Model Hyperparameters:**

  ```python
  K_factors = 30      # Number of latent factors
  n_Epochs = 100      # Number of training epochs
  lambda_reg = 1      # Regularization parameter for latent factors
  gamma = 0.01        # Regularization parameter for biases
  taw = 10            # Learning rate or regularization term
  ```

### Training Workflow

1. **Setup Logging and Experiment Folder**

   - Initializes logging to track progress.
   - Creates an experiment folder to save models and logs.

2. **Data Loading**

   - Loads preprocessed training and test data.
   - Loads index mappings for users and movies.

3. **Model Initialization**

   - Initializes user and movie latent factors with random values.
   - Initializes user and item biases with random values.

4. **Training Loop**

   For each epoch:

   - **User Updates:**

     - Updates user biases and latent factors using `Update_user_biases` and `Update_user_factors`.

   - **Movie Updates:**

     - Updates item biases and latent factors using `Update_movie_biases` and `Update_movie_factors`.

   - **Metric Calculation:**

     - Calculates training and validation loss and RMSE using `calc_metrics`.

   - **Logging and Saving:**

     - Logs the metrics and training times.
     - Saves the model parameters.

5. **Post-Training**

   - Plots and saves the loss and RMSE curves using `plot_likelihood` and `plot_rmse`.

### Functions Overview

#### Factors_Update_Functions.py

- **Update_user_biases**
- **Update_user_factors**
- **Update_movie_biases**
- **Update_movie_factors**
- **calc_metrics**

These functions perform the core computations for updating model parameters and calculating metrics.

#### Helper_Functions.py

- **plot_likelihood**
- **plot_rmse**
- **setup_logging**
- **setup_experiment_folder**
- **save_model**
- **load_model**
- **Load_training_data**
- **Load_test_data**
- **Load_idx_maps**

These utility functions assist with logging, plotting, data loading, and model saving/loading.

---

## Evaluation

### Metrics

- **Root Mean Square Error (RMSE):**

  \[
  \text{RMSE} = \sqrt{ \frac{1}{N} \sum_{(u, i)} \left( r_{ui} - \hat{r}_{ui} \right)^2 }
  \]

  Where:

  - \( N \): Total number of ratings.
  - \( r_{ui} \): Actual rating.
  - \( \hat{r}_{ui} \): Predicted rating.

### Calculation

- The `calc_metrics` function computes RMSE and total loss for both training and validation datasets after each epoch.

### Logging

- Metrics are logged to both the console and a log file in the experiment folder.
- Training and validation RMSE and loss are tracked over epochs.

### Plotting

- Loss and RMSE curves are plotted and saved to visualize training progress.

---

## Usage

### Generating Recommendations

Use the `Recommendation.py` script to generate movie recommendations based on a user's favorite movie.

#### Steps:

1. **Load Trained Model and Data**

   The script loads the trained model parameters and mappings.

2. **Specify Input Movie**

   Modify the script to specify the movie the user likes:

   ```python
   movie_name, movie_id, genre = get_movie_details(movies, "Toy Story (1995)")
   ```

3. **Create a User Profile**

   A "fake user" is created based on the favorite movie using `create_fake_user`.

4. **Generate Recommendations**

   - Computes recommendation scores by projecting the user's latent factors onto all movie factors.
   - Adjusts scores with item biases.

5. **Display Recommendations**

   Outputs a list of recommended movies and their genres, excluding the input movie.

#### Running the Script

```bash
python Recommendation.py
```

#### Sample Output

```
The user liked the movie: Toy Story (1995), with id: 1, The genre is Animation|Children|Comedy

Recommendations are:

                     title                        genre
0        Toy Story 2 (1999)  Adventure|Animation|Children|Comedy
1        Finding Nemo (2003)       Adventure|Animation|Children
2      Monsters, Inc. (2001)       Adventure|Animation|Children
...
```

### Visualizing Embeddings

Use the `Visualization.py` script to visualize movie embeddings.

#### Steps:

1. **Load Model and Mappings**

   Loads movie factors and index-to-title mappings.

2. **Dimensionality Reduction**

   Applies PCA to reduce movie latent factors to 2D for visualization.

3. **Plot Embeddings**

   Plots the reduced embeddings and labels select movies.

#### Running the Script

```bash
python Visualization.py
```

#### Sample Output

- A scatter plot displaying movies in the latent feature space, potentially revealing clusters of similar movies.

---

## Results

### Training and Validation Metrics

- **Loss and RMSE Curves:** Plots show how loss and RMSE decrease over epochs, indicating model learning.

### Recommendations

- The system provides relevant movie recommendations based on user preferences.
- Demonstrates the model's ability to capture latent similarities.

### Embeddings Visualization

- Visualizations may reveal clusters of movies with similar genres or themes.
- Helps in interpreting the learned latent features.

---

## Visualization

All plots generated during training and evaluation are saved in the respective experiment folder under `Experiments/`.

- **Loss Curves:** `Experiments/<experiment_name>/_Negative Log Likelihood Curves_experiment_plot.pdf`
- **RMSE Curves:** `Experiments/<experiment_name>/_RMSE Curves_experiment_plot.pdf`
- **Embeddings Visualization:** Saved as images when running `Visualization.py`.

---

## References

- **MovieLens Dataset:** [MovieLens 32M Dataset](https://grouplens.org/datasets/movielens/32m/)
- **Matrix Factorization Techniques:** Koren, Y., Bell, R., & Volinsky, C. (2009). [Matrix Factorization Techniques for Recommender Systems](https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf).
- **Numba Documentation:** [Numba - JIT Compiler for Python](http://numba.pydata.org/)
- **Polars Documentation:** [Polars - Fast DataFrames in Rust and Python](https://www.pola.rs/)

---

## Acknowledgments

- **GroupLens Research:** For providing the MovieLens dataset.
# Movie Recommendation System using Collaborative Filtering

## Introduction

This project implements a movie recommendation system using collaborative filtering techniques, specifically matrix factorization. The system is designed to predict user preferences for movies based on historical ratings and to generate personalized movie recommendations.

By leveraging latent factor models, the system captures underlying patterns in user behavior and movie characteristics. The project uses the MovieLens dataset, which provides a rich source of user ratings and movie metadata.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Methodology](#methodology)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Usage](#usage)
  - [Generating Recommendations](#generating-recommendations)
  - [Visualizing Embeddings](#visualizing-embeddings)
- [Results](#results)
- [Visualization](#visualization)
- [References](#references)
- [Acknowledgments](#acknowledgments)

---

## Project Overview

The goal of this project is to build a scalable and efficient movie recommendation system using collaborative filtering. The key features include:

- **Data Preprocessing:** Handling large datasets (e.g., 32 million ratings) efficiently.
- **Model Training:** Implementing matrix factorization with bias terms and regularization.
- **Performance Optimization:** Utilizing techniques like Numba for just-in-time compilation to speed up computations.
- **Evaluation:** Tracking training and validation metrics, including loss and RMSE.
- **Recommendation Generation:** Providing personalized movie recommendations based on user input.
- **Visualization:** Visualizing movie embeddings to understand the latent feature space.

---

## Project Structure

The project is organized into the following main components:

- **Scripts:**
  - `main.py`: The main script that runs the training loop for the recommendation system.
  - `Data_Preprocessing.py`: Preprocesses the dataset and prepares training and test data.
  - `Recommendation.py`: Generates movie recommendations for a user based on their favorite movies.
  - `Visualization.py`: Visualizes movie embeddings using dimensionality reduction techniques.
- **Modules:**
  - `Factors_Update_Functions.py`: Contains functions for updating user and movie biases and latent factors.
  - `Helper_Functions.py`: Utility functions for plotting, logging, saving models, and loading data.
- **Data Directories:**
  - `ml-32m/`: Contains the original MovieLens dataset files (`ratings.csv`, `movies.csv`).
  - `TRAIN_TEST_DATA/`: Stores preprocessed training and test data.
- **Experiments Directory:**
  - `Experiments/`: Contains experiment logs, models, results, and plots.
- **Miscellaneous:**
  - `README.md`: This detailed readme file.
  - `requirements.txt`: Lists required Python packages.

---

## Installation

### Prerequisites

Ensure you have the following installed:

- Python 3.7 or higher
- [NumPy](https://numpy.org/)
- [Pandas](https://pandas.pydata.org/)
- [Numba](http://numba.pydata.org/)
- [Matplotlib](https://matplotlib.org/)
- [scikit-learn](https://scikit-learn.org/)
- [Polars](https://www.pola.rs/)
- [Pickle](https://docs.python.org/3/library/pickle.html)

### Steps

1. **Clone the Repository**

   ```bash
   git clone https://github.com/your_username/your_repository.git
   cd your_repository
   ```

2. **Set Up a Virtual Environment (Recommended)**

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install Required Packages**

   Install the required packages using `pip`:

   ```bash
   pip install -r requirements.txt
   ```

   If `requirements.txt` is not available, install packages individually:

   ```bash
   pip install numpy pandas numba matplotlib scikit-learn polars
   ```

---

## Data Preparation

### Dataset

The project uses the [MovieLens 25M Dataset](https://grouplens.org/datasets/movielens/25m/), which contains millions of user ratings for movies along with movie metadata.

**Required Files:**

- `ratings.csv`: Contains user ratings (`userId`, `movieId`, `rating`, `timestamp`).
- `movies.csv`: Contains movie metadata (`movieId`, `title`, `genres`).

### Steps to Prepare Data

1. **Download the Dataset**

   Download `ratings.csv` and `movies.csv` from the MovieLens website.

2. **Place the Files**

   Place the downloaded files in the `ml-32m/` directory within the project folder.

3. **Run Data Preprocessing**

   Execute the data preprocessing script:

   ```bash
   python Data_Preprocessing.py
   ```

   This script performs the following tasks:

   - Reads and processes the ratings and movies data.
   - Maps user and movie IDs to continuous indices starting from 0 for efficient storage and computation.
   - Splits the data into training and test sets (e.g., 80% training, 20% test).
   - Groups ratings by users and movies.
   - Saves the preprocessed data and mappings to the `TRAIN_TEST_DATA/` directory.

**Note:** The preprocessing step may take some time due to the size of the dataset.

---

## Methodology

### Collaborative Filtering and Matrix Factorization

The recommendation system uses **collaborative filtering**, which makes predictions about a user's interests by collecting preferences from many users. Specifically, it employs **matrix factorization** to discover latent features underlying the interactions between users and items (movies).

### Latent Factor Model

- **User Factors (`users_factors`):** Represents the latent preferences of users.
- **Movie Factors (`movies_factors`):** Represents the latent characteristics of movies.
- **Genre Factors (`genre_factors`):** Captures genre-specific characteristics.

### Bias Terms

- **User Bias (`user_bias`):** Captures the tendency of a user to rate items higher or lower than average.
- **Item Bias (`item_bias`):** Captures the inherent popularity or quality of an item.

### Objective Function

The model minimizes the regularized squared error between predicted and actual ratings:

$
\min_{U, V, b_u, b_i} \sum_{(u, i) \in \text{Ratings}} \left( r_{ui} - (\mathbf{u}_u^\top \mathbf{v}_i + b_u + b_i) \right)^2 + \lambda \left( \|\mathbf{u}_u\|^2 + \|\mathbf{v}_i\|^2 \right) + \gamma \left( b_u^2 + b_i^2 \right)
$

Where:

- \( r_{ui} \): Actual rating of user \( u \) for item \( i \).
- \( \mathbf{u}_u \): Latent factor vector for user \( u \).
- \( \mathbf{v}_i \): Latent factor vector for item \( i \).
- \( b_u \), \( b_i \): User and item biases.
- \( \lambda \), \( \gamma \): Regularization parameters.

### Optimization Technique

- **Alternating Least Squares (ALS):** The optimization alternates between fixing user factors and updating item factors, and vice versa.
- **Regularization:** Prevents overfitting by penalizing large latent factors and biases.

### Acceleration with Numba

- **Numba JIT Compilation:** The `@jit(nopython=True)` decorator is used to compile functions to machine code at runtime, significantly speeding up computations.

---

## Model Training

### Running the Training Script

Execute the training script:

```bash
python main.py
```

### Configurable Parameters

Within `main.py`, you can adjust the following hyperparameters:

- **Experiment Settings:**

  ```python
  experiment_name = "your_experiment_name"
  ```

- **Model Hyperparameters:**

  ```python
  K_factors = 30      # Number of latent factors
  n_Epochs = 100      # Number of training epochs
  lambda_reg = 1      # Regularization parameter for latent factors
  gamma = 0.01        # Regularization parameter for biases
  taw = 10            # Learning rate or regularization term
  ```

### Training Workflow

1. **Setup Logging and Experiment Folder**

   - Initializes logging to track progress.
   - Creates an experiment folder to save models and logs.

2. **Data Loading**

   - Loads preprocessed training and test data.
   - Loads index mappings for users and movies.

3. **Model Initialization**

   - Initializes user and movie latent factors with random values.
   - Initializes user and item biases with random values.

4. **Training Loop**

   For each epoch:

   - **User Updates:**

     - Updates user biases and latent factors using `Update_user_biases` and `Update_user_factors`.

   - **Movie Updates:**

     - Updates item biases and latent factors using `Update_movie_biases` and `Update_movie_factors`.

   - **Metric Calculation:**

     - Calculates training and validation loss and RMSE using `calc_metrics`.

   - **Logging and Saving:**

     - Logs the metrics and training times.
     - Saves the model parameters.

5. **Post-Training**

   - Plots and saves the loss and RMSE curves using `plot_likelihood` and `plot_rmse`.

### Functions Overview

#### Factors_Update_Functions.py

- **Update_user_biases**
- **Update_user_factors**
- **Update_movie_biases**
- **Update_movie_factors**
- **calc_metrics**

These functions perform the core computations for updating model parameters and calculating metrics.

#### Helper_Functions.py

- **plot_likelihood**
- **plot_rmse**
- **setup_logging**
- **setup_experiment_folder**
- **save_model**
- **load_model**
- **Load_training_data**
- **Load_test_data**
- **Load_idx_maps**

These utility functions assist with logging, plotting, data loading, and model saving/loading.

---

## Evaluation

### Metrics

- **Root Mean Square Error (RMSE):**

  \[
  \text{RMSE} = \sqrt{ \frac{1}{N} \sum_{(u, i)} \left( r_{ui} - \hat{r}_{ui} \right)^2 }
  \]

  Where:

  - \( N \): Total number of ratings.
  - \( r_{ui} \): Actual rating.
  - \( \hat{r}_{ui} \): Predicted rating.

### Calculation

- The `calc_metrics` function computes RMSE and total loss for both training and validation datasets after each epoch.

### Logging

- Metrics are logged to both the console and a log file in the experiment folder.
- Training and validation RMSE and loss are tracked over epochs.

### Plotting

- Loss and RMSE curves are plotted and saved to visualize training progress.

---

## Usage

### Generating Recommendations

Use the `Recommendation.py` script to generate movie recommendations based on a user's favorite movie.

#### Steps:

1. **Load Trained Model and Data**

   The script loads the trained model parameters and mappings.

2. **Specify Input Movie**

   Modify the script to specify the movie the user likes:

   ```python
   movie_name, movie_id, genre = get_movie_details(movies, "Toy Story (1995)")
   ```

3. **Create a User Profile**

   A "fake user" is created based on the favorite movie using `create_fake_user`.

4. **Generate Recommendations**

   - Computes recommendation scores by projecting the user's latent factors onto all movie factors.
   - Adjusts scores with item biases.

5. **Display Recommendations**

   Outputs a list of recommended movies and their genres, excluding the input movie.

#### Running the Script

```bash
python Recommendation.py
```

#### Sample Output

```
The user liked the movie: Toy Story (1995), with id: 1, The genre is Animation|Children|Comedy

Recommendations are:

                     title                        genre
0        Toy Story 2 (1999)  Adventure|Animation|Children|Comedy
1        Finding Nemo (2003)       Adventure|Animation|Children
2      Monsters, Inc. (2001)       Adventure|Animation|Children
...
```

### Visualizing Embeddings

Use the `Visualization.py` script to visualize movie embeddings.

#### Steps:

1. **Load Model and Mappings**

   Loads movie factors and index-to-title mappings.

2. **Dimensionality Reduction**

   Applies PCA to reduce movie latent factors to 2D for visualization.

3. **Plot Embeddings**

   Plots the reduced embeddings and labels select movies.

#### Running the Script

```bash
python Visualization.py
```

#### Sample Output

- A scatter plot displaying movies in the latent feature space, potentially revealing clusters of similar movies.

---

## Results

### Training and Validation Metrics

- **Loss and RMSE Curves:** Plots show how loss and RMSE decrease over epochs, indicating model learning.

### Recommendations

- The system provides relevant movie recommendations based on user preferences.
- Demonstrates the model's ability to capture latent similarities.

### Embeddings Visualization

- Visualizations may reveal clusters of movies with similar genres or themes.
- Helps in interpreting the learned latent features.

---

## Visualization

All plots generated during training and evaluation are saved in the respective experiment folder under `Experiments/`.

- **Loss Curves:** `Experiments/<experiment_name>/_Negative Log Likelihood Curves_experiment_plot.pdf`
- **RMSE Curves:** `Experiments/<experiment_name>/_RMSE Curves_experiment_plot.pdf`
- **Embeddings Visualization:** Saved as images when running `Visualization.py`.

---

## References

- **MovieLens Dataset:** [MovieLens 32M Dataset](https://grouplens.org/datasets/movielens/32m/)
- **Matrix Factorization Techniques:** Koren, Y., Bell, R., & Volinsky, C. (2009). [Matrix Factorization Techniques for Recommender Systems](https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf).
- **Numba Documentation:** [Numba - JIT Compiler for Python](http://numba.pydata.org/)
- **Polars Documentation:** [Polars - Fast DataFrames in Rust and Python](https://www.pola.rs/)

---

## Acknowledgments

- **GroupLens Research:** For providing the MovieLens dataset.
