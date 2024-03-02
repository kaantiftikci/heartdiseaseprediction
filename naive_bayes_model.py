import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import GaussianNB
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score


df = pd.read_csv("/Users/kaantiftikci/Desktop/framingham.csv")

df["education"].fillna(df["education"].median(), inplace=True)
df["cigsPerDay"].fillna(df["cigsPerDay"].median(), inplace=True)
df["BPMeds"].fillna(df["BPMeds"].median(), inplace=True)
df["totChol"].fillna(df["totChol"].median(), inplace=True)
df["BMI"].fillna(df["BMI"].median(), inplace=True)
df["heartRate"].fillna(df["heartRate"].median(), inplace=True)
df["glucose"].fillna(df["glucose"].median(), inplace=True)


X = df.iloc[:, :-1]
y = df.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

parameter_grid = {'var_smoothing': np.logspace(0,-9, num=100)}
naive_bayes_classifier = GaussianNB()
grid_search = GridSearchCV(naive_bayes_classifier, parameter_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
joblib.dump(best_model, 'heart_prediction_naive_bayes_optimized.joblib')

y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Optimized Model Accuracy:", accuracy)
print("\nOptimized Classification Report:")

print(classification_report(y_test, y_pred))
print("\nOptimized Confusion Matrix:")

print(confusion_matrix(y_test, y_pred))
print("\nOptimized AUC-ROC Value:", roc_auc_score(y_test, y_pred))

