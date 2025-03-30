# PACKAGES
import numpy as np
import requests
import matplotlib.pyplot as plt
from xgboost import XGBClassifier

from sklearn.datasets import make_classification

from sklearn.impute import SimpleImputer
from joblib import parallel_backend

from sklearn.model_selection import cross_val_score

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_fscore_support as score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, cross_val_score, train_test_split, RandomizedSearchCV


import pandas as pd
import seaborn as sns

import requests


# IMPORT DATASET
import requests
url_dict = {
    'train.csv': 'http://drive.google.com/uc?export=download&id=1GhQRifwbBjX9sOFFJClT1_ECEVpXiVD5',
    'test.csv': 'http://drive.google.com/uc?export=download&id=12ykDwsNAq1rkYslQD4eIQKfoRD88NsdI',
}



def download_file(file_path):
    url = url_dict[file_path]
    print('Start downloading...')
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(file_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024 * 1024):
                f.write(chunk)
    print('Complete')

def load_data(filename):
    """
    Function loads data stored in the file filename and returns it as a numpy ndarray.

    Inputs:
        filename: GeneratorExitiven as a string.

    Outputs:
        Data contained in the file, returned as a numpy ndarray
    """
    return pd.read_csv(filename)


download_file('train.csv')
train_set = load_data('train.csv')
download_file('test.csv')
test_set = load_data('test.csv')

print(train_set.head())

# CLEAN DATA

# Train Set
train_set['release_year'] = train_set['track_album_release_date'].str.slice(0, 4)
train_set['release_month'] = train_set['track_album_release_date'].str.slice(5, 7)
train_set['release_day'] = train_set['track_album_release_date'].str.slice(8, 10)
train_set['release_year'] = pd.to_numeric(train_set['release_year'])
train_set['release_month'] = pd.to_numeric(train_set['release_month'])
train_set['release_day'] = pd.to_numeric(train_set['release_day'])

# 'High' = 1, 'Low' = 0
train_set['Popularity_Type'] = train_set['Popularity_Type'] == 'High'
train_set['Popularity_Type'] = train_set['Popularity_Type'].astype(int)

train_set = train_set.drop(['track_href', 'uri', 'type', 'analysis_url', 'track_album_release_date'], axis='columns')
print(train_set.columns)

# Test Set
test_set['release_year'] = test_set['track_album_release_date'].str.slice(0, 4)
test_set['release_month'] = test_set['track_album_release_date'].str.slice(5, 7)
test_set['release_day'] = test_set['track_album_release_date'].str.slice(8, 10)
test_set['release_year'] = pd.to_numeric(test_set['release_year'])
test_set['release_month'] = pd.to_numeric(test_set['release_month'])
test_set['release_day'] = pd.to_numeric(test_set['release_day'])

test_set = test_set.drop(['track_href', 'uri', 'type', 'analysis_url', 'track_album_release_date'], axis='columns')
print(test_set.columns)


# NORMALIZE DATA

# Training Set

train_set['time_signature'] = train_set['time_signature'] / max(train_set['time_signature'])
train_set['duration_ms'] = train_set['duration_ms'] / max(train_set['duration_ms'])
train_set['release_year'] = train_set['release_year'] / max(train_set['release_year'])
train_set['release_month'] = train_set['release_month'] / max(train_set['release_month'])
train_set['release_day'] = train_set['release_day'] / max(train_set['release_day'])
train_set['key'] = train_set['key'] / max(train_set['key'])
train_set['tempo'] = train_set['tempo'] / max(train_set['tempo'])
train_set['loudness'] = train_set['loudness'] - min(train_set['loudness'])
train_set['loudness'] = train_set['loudness'] / max(train_set['loudness'])

# Test Set
test_set['time_signature'] = test_set['time_signature'] / max(test_set['time_signature'])
test_set['duration_ms'] = test_set['duration_ms'] / max(test_set['duration_ms'])
test_set['release_year'] = test_set['release_year'] / max(test_set['release_year'])
test_set['release_month'] = test_set['release_month'] / max(test_set['release_month'])
test_set['release_day'] = test_set['release_day'] / max(test_set['release_day'])
test_set['key'] = test_set['key'] / max(test_set['key'])
test_set['tempo'] = test_set['tempo'] / max(test_set['tempo'])
test_set['loudness'] = test_set['loudness'] - min(test_set['loudness'])
test_set['loudness'] = test_set['loudness'] / max(test_set['loudness'])


# CHECK IF HIGH'S AND LOW'S ARE ABOUT 50% EACH
print(sum(train_set['Popularity_Type'] == 1))
print(sum(train_set['Popularity_Type'] == 0))

# VISUALIZE DATA
train_set =train_set.drop('id', axis='columns')
corr_matrix = train_set.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", xticklabels=corr_matrix.columns, yticklabels=corr_matrix.columns)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


def visualize_dataset(X, Y, title):
    # Set colormap such that blue is positive and red is negative.
    plt.close('all')
    plt.set_cmap('bwr')
    plt.figure(figsize=(6, 5))

    # Plot data.
    plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=Y, s=10)
    plt.title(title)
    plt.colorbar()
    plt.show()

visualize_dataset(train_set.iloc[:, [1, 2]], train_set.iloc[:, -1], "Speechiness vs Dancibility")

visualize_dataset(train_set.iloc[:, [1, 4]], train_set.iloc[:, -1], "Speechiness vs Energy")

visualize_dataset(train_set.iloc[:, [2, 4]], train_set.iloc[:, -1], "Dancebility vs Energy")

visualize_dataset(train_set.iloc[:, [12, 15]], train_set.iloc[:, -1], "Instrumentalness vs Tempo")

visualize_dataset(train_set.iloc[:, [1, 12]], train_set.iloc[:, -1], "Instrumentalness vs Tempo")



# XGBOOST TREE OR RANDOM FOREST

# scaler
scaler = StandardScaler()
imputer = SimpleImputer(strategy='mean')


X = train_set.drop(['Popularity_Type'], axis='columns')
y = train_set['Popularity_Type']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_train = imputer.fit_transform(X_train)
X_train = scaler.fit_transform(X_train)
X_test = imputer.transform(X_test)
X_test = scaler.transform(X_test)


# WITH OVERFITTING
models = {
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "Logistic Regression": LogisticRegression(),
    "XGBoost": XGBClassifier(),
    "KNN": KNeighborsClassifier(),
    "Neural Network": MLPClassifier(max_iter=500, early_stopping=True, verbose=0),
    "SVM": SVC(),
    "Linear Regression": LinearRegression()
}
for name, model in models.items():
    model.fit(X_train, y_train)
    # print(name +" trained")
for name, model in models.items():
    print(name + " test score: {:.2f}%".format(model.score(X_test, y_test) * 100))
    print(name + " train score: {:.2f}%".format(model.score(X_train, y_train) * 100))



# PREVENTING OVERFITTING
models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42,
        min_samples_split=2,
        min_samples_leaf=1
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    ),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=5,
        random_state=42,
        min_samples_split=2,
        min_samples_leaf=1
    ),
    "Logistic Regression": LogisticRegression(
        C=1.0,
        penalty='l2',
        random_state=42
    ),
    "XGBoost": XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    ),
    "KNN": KNeighborsClassifier(
      n_neighbors=5
    ),
    "Neural Network": MLPClassifier(
        max_iter=500,
        early_stopping=True,
        verbose=0,
        hidden_layer_sizes=(100, 50)
    ),
    "SVM": SVC(
        C=1.0,
        kernel='rbf',
        random_state=42
    ),
    "Linear Regression": LinearRegression()
}
for name, model in models.items():
    model.fit(X_train, y_train)
    print(name +" trained")


# RESULTS
for name, model in models.items():
    print(name + " test score: {:.2f}%".format(model.score(X_test, y_test) * 100))
    print(name + " train score: {:.2f}%".format(model.score(X_train, y_train) * 100))



# CREATING SUBMISSION.CSV
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"training set has {X_train.shape[0]} samples")
print(f"testing set has {X_test.shape[0]} samples")

X_train_transformed = pd.DataFrame(
    scaler.fit_transform(X_train),
    columns=X_train.columns,
    index=X_train.index)
X_train_transformed = imputer.fit_transform(X_train_transformed)

X_val_transformed = pd.DataFrame(
    scaler.transform(X_test),
    columns=X_test.columns,
    index=X_test.index)
X_val_transformed = imputer.transform(X_val_transformed)

X_test_transformed = test_set.drop(['ID'], axis='columns')
X_test_transformed = pd.DataFrame(
    scaler.transform(X_test_transformed),
    columns=X_test_transformed.columns,
    index=X_test_transformed.index)

X_test_transformed = imputer.transform(X_test_transformed)

xgboost = XGBClassifier(
    random_state=42,
    eval_metric='auc',  # Use 'auc' for evaluation
    early_stopping_rounds=10,  # Stop if no improvement in 10 rounds
    scale_pos_weight= float(np.sum(y_train == 0)) / np.sum(y_train == 1)  # Handle class imbalance
)

param_grid = {
    'max_depth': [3, 5],  # Controls the depth of the trees
    'learning_rate': [0.1, 0.2],  # Step size shrinkage
    'n_estimators': [100, 200],   # Number of boosting rounds
    'subsample': [0.8, 1.0],  # Fraction of samples used for training
    'colsample_bytree': [0.8, 1.0],  # Fraction of features used for training
    'gamma': [0, 0.1],  # Minimum loss reduction to make a split
    'reg_alpha': [0, 0.1],  # L1 regularization
    'reg_lambda': [0, 0.1]  # L2 regularization
}


grid_search = GridSearchCV(
    estimator=xgboost,
    param_grid=param_grid,
    scoring='roc_auc',  # Use AUC for evaluation
    refit='recall',  # Refit the model with the best parameters
    cv=5,  # 5-fold cross-validation
    n_jobs=-1,
    verbose=1,
)

grid_search.fit(X_train_transformed, y_train, eval_set=[(X_val_transformed, y_test)])

best_params = grid_search.best_params_
print("Best parameters:", best_params)
best_model = grid_search.best_estimator_
y_pred_test = best_model.predict(X_val_transformed)
print( " test score: {:.2f}%".format(best_model.score(X_val_transformed, y_pred_test) * 100))

y_pred_final = best_model.predict_proba(X_test_transformed)[:, 1]

submission = pd.DataFrame({'ID': test_set['ID'], 'Popularity_Type': y_pred_final})
submission.to_csv('submission.csv', index=False)
