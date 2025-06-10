import json
import dagshub
import mlflow
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, log_loss, roc_auc_score, f1_score, recall_score, \
    confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns

# Init dagshub
dagshub.init(repo_owner='refandasuryasaputra', repo_name='mushroom-model', mlflow=True)

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

    # Metrik tambahan
    test_log_loss = log_loss(y_test, y_pred)
    test_roc_auc = roc_auc_score(y_test, y_pred)

    # Mencatat metrik training dan testing ke MLFlow
    mlflow.log_metrics({
        'train_accuracy': train_accuracy,
        'train_precision': train_precision,
        'train_recall': train_recall,
        'train_f1': train_f1,
        'train_log_loss': train_log_loss,
        'train_roc_auc': train_roc_auc,
        'test_accuracy': test_accuracy,
        'test_log_loss': test_log_loss,
        'test_roc_auc': test_roc_auc,
    })

    # Menyimpan model terbaik ke MLFlow
    mlflow.sklearn.log_model(best_model, 'model', input_example=input_example)

    # Menyimpan metric info dalam bentuk json
    metric_info = {
        'LogisticRegression_score_X_test': 'LogisticRegression.score(X=X_test, y=y_test)',
    }

    metric_info_path = "metric_info.json"

    with open(metric_info_path, "w") as f:
        json.dump(metric_info, f, indent=4)

    mlflow.log_artifact(metric_info_path)

    # Menyimpan confusion matrix training
    cm_train = confusion_matrix(y_train, y_train_pred)

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(cm_train, annot=True, fmt='g', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion matrix')

    cm_train_path = 'training_confusion_matrix.png'

    plt.savefig(cm_train_path)
    plt.close(fig)

    mlflow.log_artifact(cm_train_path)

    # Menyimpan parameter model sebagai html
    estimator_html = f"""
        <!doctype html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <title>Document</title>
        </head>
        <body>
            <h1>Detail Model Estimator</h1>
            <p>Rangkuman Model Regresi Logistik yang telah dilatih</p>
            <h2>Parameters</h2>
            {best_params}
        </body>
        </html>
    """

    estimator_html_path = "estimator.html"

    with open(estimator_html_path, "w") as f:
        f.write(estimator_html)

    mlflow.log_artifact(estimator_html_path)