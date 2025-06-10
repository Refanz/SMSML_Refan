import mlflow
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, log_loss, roc_auc_score, f1_score, recall_score
from sklearn.model_selection import train_test_split, GridSearchCV

mlflow.set_tracking_uri("http://127.0.0.1:5000/")
mlflow.set_experiment("SMSML")

# Load dataset dari github
dataset_url = "https://raw.githubusercontent.com/refanz/Eksperimen_SML_Refan/master/preprocessing/mushrooms_preprocessed.csv"
mushroom_df = pd.read_csv(dataset_url)

# Melakukan splitting dataset
X = mushroom_df.drop(columns=['class'], axis=1)
y = mushroom_df['class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Menyimpan sample input
input_example = X_train[0:5]

# Mendifiisikan parameter untuk metode Grid Search
param_grid = {
    'penalty': ['l1', 'l2', 'elasticnet'],
    'solver': ['saga', 'liblinear'],
    'C': [0.001, 0.01, 0.1, 1.0, 10],
}

with mlflow.start_run():
    grid_search = GridSearchCV(
        estimator=LogisticRegression(random_state=42),
        param_grid=param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=1,
    )

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    # Manual logging untuk best parameter
    mlflow.log_params(best_params)

    # Manual logging untuk skor metrik saat training model
    y_train_pred = best_model.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_precision = precision_score(y_train, y_train_pred, average='binary')
    train_recall = recall_score(y_train, y_train_pred, average='binary')
    train_f1 = f1_score(y_train, y_train_pred, average='binary')
    train_roc_auc = roc_auc_score(y_train, y_train_pred)
    train_log_loss = log_loss(y_train, y_train_pred)

    # Manual logging untuk akurasi saat testing model
    y_pred = best_model.predict(X_test)
    test_accuracy = best_model.score(X_test, y_test)

    # Mencatat metrik training dan testing ke MLFlow
    mlflow.log_metrics({
        'train_accuracy': train_accuracy,
        'train_precision': train_precision,
        'train_recall': train_recall,
        'train_f1': train_f1,
        'train_log_loss': train_log_loss,
        'train_roc_auc': train_roc_auc,
        'test_accuracy': test_accuracy,
    })

    # Menyimpan model terbaik ke MLFlow
    mlflow.sklearn.log_model(best_model, 'model', input_example=input_example)





