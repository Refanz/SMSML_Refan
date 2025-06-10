import mlflow
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

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

# Melatih model dan menjalankan mlflow
with mlflow.start_run():
    # Mengaktifkan autologging
    mlflow.autolog()

    model = LogisticRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    test_score = model.score(X_test, y_test)

    print(f"Accuracy Score Test: {test_score}")
