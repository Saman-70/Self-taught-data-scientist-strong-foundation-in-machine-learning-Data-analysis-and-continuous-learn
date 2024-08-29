# Self-taught-data-scientist-strong-foundation-in-machine-learning-Data-analysis-and-continuous-learning.
Projects:- House prices using advance regression technique from kaggle.

Develop a regression machine learning model to predict the house prices in diffrent location with help of csv file consisting of all data of houses including number of bedrooms, bathroom, kitchen size, garage area in square foot, plot in swuare foot and also evaluate the model using mean squared error, mean absolute error.

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load dataset
# df = pd.read_csv('house_prices.csv')

# Preprocessing
# X = df.drop('price', axis=1)
# y = df['price']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model pipeline
# numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
# categorical_features = X.select_dtypes(include=['object']).columns
# preprocessor = ColumnTransformer(
#    transformers=[
#        ('num', StandardScaler(), numeric_features),
#        ('cat', OneHotEncoder(), categorical_features)
#    ])

# pipeline = Pipeline(steps=[('preprocessor', preprocessor),
#                            ('regressor', RandomForestRegressor())])

# Hyperparameter tuning
# param_grid = {'regressor__n_estimators': [100, 200], 'regressor__max_depth': [10, 20]}
# grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error')
# grid_search.fit(X_train, y_train)

# Evaluate
# y_pred = grid_search.predict(X_test)
# mse = mean_squared_error(y_test, y_pred)
# print(f'Mean Squared Error: {mse}')
Movie Recommendations system:- 

1. Multiple Linear Regression
Concept: Predict the price of a house using multiple independent variables (e.g., size, number of bedrooms, location).
Advantages: Simple to implement and interpret.
Disadvantages: Assumes a linear relationship between the independent variables and the target variable (house price).
2. Polynomial Regression
Concept: Extends linear regression by considering polynomial relationships (e.g., quadratic or cubic) between the independent variables and the target.
Advantages: Can model more complex relationships.
Disadvantages: Can overfit if the degree of the polynomial is too high.
3. Ridge Regression (L2 Regularization)
Concept: A type of linear regression that includes a penalty for large coefficients to prevent overfitting.
Advantages: Helps to manage multicollinearity, improves generalization.
Disadvantages: Requires tuning the regularization parameter.
4. Lasso Regression (L1 Regularization)
Concept: Similar to Ridge Regression but with L1 penalty, which can shrink coefficients to zero, effectively selecting a subset of features.
Advantages: Performs feature selection automatically.
Disadvantages: Can be sensitive to outliers.
5. Elastic Net
Concept: Combines L1 and L2 regularization techniques.
Advantages: Balances the benefits of Ridge and Lasso, good for datasets with many correlated features.
Disadvantages: Requires tuning two parameters.
6. Decision Trees
Concept: Uses a tree-like structure to model decisions and their possible consequences.
Advantages: Handles non-linear relationships well, easy to interpret.
Disadvantages: Can overfit, especially with deep trees.
7. Random Forest
Concept: An ensemble method that uses multiple decision trees and aggregates their predictions.
Advantages: Reduces overfitting, works well with non-linear data.
Disadvantages: Can be computationally expensive.
8. Gradient Boosting (e.g., XGBoost, LightGBM)
Concept: Sequentially builds models that correct the errors of the previous models.
Advantages: Often provides the best performance in prediction tasks.
Disadvantages: More complex to implement, requires careful tuning of hyperparameters.
9. Support Vector Regression (SVR)
Concept: A type of regression that uses a boundary to maximize the margin of error allowed.
Advantages: Effective in high-dimensional spaces.
Disadvantages: Computationally intensive, requires feature scaling.
10. Neural Networks
Concept: Uses layers of neurons to model complex patterns.
Advantages: Can model highly non-linear relationships, excellent for large datasets.
Disadvantages: Requires large amounts of data, computationally expensive, difficult to interpret.
Steps to Implement:
Data Preprocessing:

Handle missing values.
Normalize or standardize features.
Encode categorical variables (e.g., using one-hot encoding).
Feature Engineering:

Create new features (e.g., interaction terms, polynomial features).
Select the most relevant features (e.g., using Lasso, feature importance from trees).
Model Selection:

Start with simpler models (e.g., linear regression) and progress to more complex ones (e.g., Gradient Boosting).
Use cross-validation to select the best model and hyperparameters.
Evaluation:

Use metrics like RMSE (Root Mean Square Error), MAE (Mean Absolute Error), or R-squared to evaluate the model.
Check for overfitting by comparing training and validation errors.
Example Workflow in Python:
python
Copy code
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load dataset
# df = pd.read_csv('house_prices.csv')

# Preprocessing
# X = df.drop('price', axis=1)
# y = df['price']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model pipeline
# numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
# categorical_features = X.select_dtypes(include=['object']).columns
# preprocessor = ColumnTransformer(
#    transformers=[
#        ('num', StandardScaler(), numeric_features),
#        ('cat', OneHotEncoder(), categorical_features)
#    ])

# pipeline = Pipeline(steps=[('preprocessor', preprocessor),
#                            ('regressor', RandomForestRegressor())])

# Hyperparameter tuning
# param_grid = {'regressor__n_estimators': [100, 200], 'regressor__max_depth': [10, 20]}
# grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error')
# grid_search.fit(X_train, y_train)

# Evaluate
# y_pred = grid_search.predict(X_test)
# mse = mean_squared_error(y_test, y_pred)
# print(f'Mean Squared Error: {mse}')
This workflow can be adapted to use any of the techniques mentioned above.
Movie Recommendations system:

1. Content-Based Filtering
Concept: Recommends movies similar to the ones the user has watched and liked based on features like genre, actors, directors, etc.

Steps:

Feature Extraction: Create a profile for each movie based on its attributes (e.g., genres, actors, directors).
Similarity Measurement: Use a similarity metric (e.g., cosine similarity) to compare movies.
Recommendation: Recommend movies similar to the ones the user has liked.
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# Sample data
movies = ['Movie A', 'Movie B', 'Movie C']
genres = ['Action|Adventure', 'Action|Sci-Fi', 'Drama|Romance']

# Convert genres to vectors
vectorizer = CountVectorizer(tokenizer=lambda x: x.split('|'))
genre_matrix = vectorizer.fit_transform(genres)

# Compute similarity
similarity_matrix = cosine_similarity(genre_matrix)

# Recommend similar movies to 'Movie A'
movie_index = 0  # Index for 'Movie A'
similar_movies = list(enumerate(similarity_matrix[movie_index]))
similar_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)

# Display recommendations
for i, score in similar_movies[1:]:
    print(f"Recommended: {movies[i]} with similarity score: {score}")
    from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Sample user-movie ratings matrix
data = {'Movie A': [5, 4, 0],
        'Movie B': [3, 0, 4],
        'Movie C': [0, 2, 3]}
user_movie_matrix = pd.DataFrame(data, index=['User 1', 'User 2', 'User 3'])

# Compute user similarity
similarity_matrix = cosine_similarity(user_movie_matrix)
similarity_df = pd.DataFrame(similarity_matrix, index=user_movie_matrix.index, columns=user_movie_matrix.index)

# Recommend for User 1
user_index = 0  # Index for 'User 1'
similar_users = list(enumerate(similarity_matrix[user_index]))
similar_users = sorted(similar_users, key=lambda x: x[1], reverse=True)

# Aggregate ratings from similar users
recommended_movies = user_movie_matrix.iloc[similar_users[1][0]].copy()
for i, score in similar_users[2:]:
    recommended_movies += user_movie_matrix.iloc[i] * score

recommended_movies = recommended_movies.sort_values(ascending=False)
print(recommended_movies)
from sklearn.decomposition import TruncatedSVD
import numpy as np

# Assuming user_movie_matrix is the same as above
user_movie_matrix.fillna(0, inplace=True)  # Replace NaNs with zeros

# Apply SVD
svd = TruncatedSVD(n_components=2)  # Choose number of latent factors
user_factors = svd.fit_transform(user_movie_matrix)
movie_factors = svd.components_.T

# Predict ratings
predicted_ratings = np.dot(user_factors, movie_factors.T)
predicted_ratings_df = pd.DataFrame(predicted_ratings, index=user_movie_matrix.index, columns=user_movie_matrix.columns)

# Recommend movies for User 1
recommendations = predicted_ratings_df.iloc[0].sort_values(ascending=False)
print(recommendations)
from keras.models import Model
from keras.layers import Input, Embedding, Flatten, Dot, Add

# Inputs
user_input = Input(shape=(1,))
movie_input = Input(shape=(1,))

# Embeddings
user_embedding = Embedding(input_dim=num_users, output_dim=latent_dim)(user_input)
movie_embedding = Embedding(input_dim=num_movies, output_dim=latent_dim)(movie_input)

user_vec = Flatten()(user_embedding)
movie_vec = Flatten()(movie_embedding)

# Dot product for similarity
dot = Dot(axes=1)([user_vec, movie_vec])

# Final model
model = Model([user_input, movie_input], dot)
model.compile(optimizer='adam', loss='mse')

# Train the model with user_movie data
model.fit([user_ids, movie_ids], ratings, epochs=20, verbose=1)

# Predict ratings
predictions = model.predict([new_user_ids, new_movie_ids])
Deployment
API Development: Develop an API using Flask, FastAPI, or Django to serve your recommendations.
Real-time Updates: Implement mechanisms to update recommendations in real-time as user preferences change.
Additional Tools:
Libraries: scikit-learn, Surprise, lightFM, TensorFlow, PyTorch.
Datasets: MovieLens, IMDb datasets for testing and development.
Churn prediction:- Understanding the Data
Common Features: Customer demographics, account information, service usage patterns, support tickets, contract type, etc.
Target Variable: Churn (typically a binary variable: 1 if the customer churned, 0 if they stayed).
Data Preprocessing
Handling Missing Values: Impute or remove missing data.
Encoding Categorical Variables: Convert categorical variables to numerical format (e.g., using one-hot encoding).
Feature Scaling: Normalize or standardize numerical features.
Feature Engineering: Create new features (e.g., tenure, monthly charges to total charges ratio).
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

# Load dataset (replace 'customer_churn.csv' with your actual dataset)
df = pd.read_csv('customer_churn.csv')

# Preprocessing: Handle missing values, encoding categorical variables, feature scaling, etc.
# Example: Label Encoding for 'gender' column (modify according to your dataset)
df['gender'] = LabelEncoder().fit_transform(df['gender'])

# Feature-target split
X = df.drop('Churn', axis=1)
y = df['Churn']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Handle Imbalanced Data using SMOTE (optional, depending on your data)
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Feature scaling
scaler = StandardScaler()
X_train_resampled = scaler.fit_transform(X_train_resampled)
X_test = scaler.transform(X_test)

# Model training using Random Forest
model = RandomForestClassifier(random_state=42)
model.fit(X_train_resampled, y_train_resampled)

# Hyperparameter Tuning using GridSearchCV (optional)
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy', verbose=2)
grid_search.fit(X_train_resampled, y_train_resampled)
print("Best parameters:", grid_search.best_params_)

# Use the best estimator from GridSearchCV
best_model = grid_search.best_estimator_

# Predictions on the test set
y_pred = best_model.predict(X_test)

# Evaluation: Accuracy, Confusion Matrix, Classification Report
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# ROC-AUC Score and ROC Curve
y_pred_proba = best_model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_pred_proba)
print("ROC-AUC Score:", auc)

fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

# Feature Importance
importances = best_model.feature_importances_
feature_names = X.columns
feature_importance = pd.Series(importances, index=feature_names).sort_values(ascending=False)
print("Feature Importance:\n", feature_importance)
