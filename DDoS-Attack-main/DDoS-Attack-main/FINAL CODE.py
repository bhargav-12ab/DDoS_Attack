FINAL CODE

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the dataset
df = pd.read_csv('dataset_sdn.csv')

# Preprocess the data
df = df[df['pktrate'] != 0]
df['switch'] = df['switch'].astype(str)
df['port_no'] = df['port_no'].astype(str)
df = pd.get_dummies(df, columns=['switch', 'Protocol'])
df = df.select_dtypes(include=[np.number])
df = df.fillna(df.mean())

# Feature selection
X = df.drop('label', axis=1)
y = df['label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

# Feature scaling
numeric_scaler = StandardScaler()
X_train_scaled = numeric_scaler.fit_transform(X_train)
X_test_scaled = numeric_scaler.transform(X_test)

# Model training and evaluation
models = {
    'GNB': GaussianNB(),
    'SVC': SVC(),
    'DT': DecisionTreeClassifier(),
    'RF': RandomForestClassifier(),
    'XGB': XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    train_score = accuracy_score(y_train, model.predict(X_train_scaled))
    test_score = accuracy_score(y_test, model.predict(X_test_scaled))
    print(f"{name}: Train score - {train_score*100:.2f}%, Test score - {test_score*100:.2f}%")
    print("Confusion matrix for {0}:\n{1}".format(name, confusion_matrix(y_test, model.predict(X_test_scaled))))
    print("Classification report for {0}:\n{1}".format(name, classification_report(y_test, model.predict(X_test_scaled))))

# User input and prediction
numeric_input_values = [float(input(f"Enter {col}: ")) for col in X.columns]

input_array_numeric = np.array(numeric_input_values).reshape(1, -1)

# Scale the numeric input using the already fitted scaler
input_array_scaled = numeric_scaler.transform(input_array_numeric)

# Make the prediction using the XGBoost model
xgb = models['XGB']
prediction = xgb.predict(input_array_scaled)[0]
result = "Benign" if prediction == 0 else "Malicious"
print(f"Predicted result: {result}")
