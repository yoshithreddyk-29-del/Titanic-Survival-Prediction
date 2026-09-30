import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

# Deep Learning
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten


# =========================================================
# 1. CREATE FOLDERS
# =========================================================

os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)


# =========================================================
# 2. LOAD DATASET
# =========================================================

df = pd.read_csv("data/Titanic-Dataset.csv")

print("\n===== DATASET HEAD =====")
print(df.head())

print("\n===== DATASET SHAPE =====")
print(df.shape)

print("\n===== DATASET INFO =====")
print(df.info())

print("\n===== MISSING VALUES =====")
print(df.isnull().sum())


# =========================================================
# 3. BASIC DATA ANALYSIS
# =========================================================

print("\n===== SURVIVAL DISTRIBUTION =====")
print(df["Survived"].value_counts())

plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="Survived")
plt.title("Survival Distribution")
plt.tight_layout()
plt.savefig("outputs/survival_distribution.png")
plt.close()


plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="Sex", hue="Survived")
plt.title("Survival by Gender")
plt.tight_layout()
plt.savefig("outputs/survival_by_gender.png")
plt.close()


plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="Pclass", hue="Survived")
plt.title("Survival by Passenger Class")
plt.tight_layout()
plt.savefig("outputs/survival_by_class.png")
plt.close()


# =========================================================
# 4. FEATURE SELECTION
# =========================================================

features = [
    "Pclass",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Embarked"
]

X = df[features]
y = df["Survived"]


# =========================================================
# 5. TRAIN-TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)


# =========================================================
# 6. DATA PREPROCESSING
# =========================================================

numeric_features = [
    "Pclass",
    "Age",
    "SibSp",
    "Parch",
    "Fare"
]

categorical_features = [
    "Sex",
    "Embarked"
]


numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])


categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])


preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])


# =========================================================
# 7. MACHINE LEARNING MODELS
# =========================================================

logistic_model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(max_iter=1000))
])


decision_tree = Pipeline([
    ("preprocessor", preprocessor),
    ("model", DecisionTreeClassifier(
        random_state=42,
        max_depth=5
    ))
])


random_forest = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=300,
        random_state=42
    ))
])


gradient_boosting = Pipeline([
    ("preprocessor", preprocessor),
    ("model", GradientBoostingClassifier(
        random_state=42
    ))
])


xgb_model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42
    ))
])


ml_models = {
    "Logistic Regression": logistic_model,
    "Decision Tree": decision_tree,
    "Random Forest": random_forest,
    "Gradient Boosting": gradient_boosting,
    "XGBoost": xgb_model
}


# =========================================================
# 8. TRAIN MACHINE LEARNING MODELS
# =========================================================

ml_results = []

print("\n===== MACHINE LEARNING RESULTS =====")

for name, model in ml_models.items():

    print("\nTraining:", name)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    results = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_prob)
    }

    ml_results.append(results)

    print(results)


ml_results_df = pd.DataFrame(ml_results)


# =========================================================
# 9. DEEP LEARNING PREPROCESSING
# =========================================================

dl_numeric = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])


dl_categorical = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])


dl_preprocessor = ColumnTransformer([
    ("num", dl_numeric, numeric_features),
    ("cat", dl_categorical, categorical_features)
])


X_train_dl = dl_preprocessor.fit_transform(X_train)
X_test_dl = dl_preprocessor.transform(X_test)

print("\nDeep Learning input shape:", X_train_dl.shape)


# =========================================================
# 10. ANN MODEL
# =========================================================

ann_model = Sequential([
    Dense(
        64,
        activation="relu",
        input_shape=(X_train_dl.shape[1],)
    ),

    Dropout(0.30),

    Dense(32, activation="relu"),

    Dropout(0.20),

    Dense(16, activation="relu"),

    Dense(1, activation="sigmoid")
])


ann_model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


print("\n===== TRAINING ANN =====")

ann_model.fit(
    X_train_dl,
    y_train,
    validation_split=0.20,
    epochs=50,
    batch_size=32,
    verbose=1
)


ann_prob = ann_model.predict(X_test_dl).ravel()

ann_pred = (ann_prob >= 0.5).astype(int)


ann_results = {
    "Model": "ANN",
    "Accuracy": accuracy_score(y_test, ann_pred),
    "Precision": precision_score(y_test, ann_pred),
    "Recall": recall_score(y_test, ann_pred),
    "F1 Score": f1_score(y_test, ann_pred),
    "ROC-AUC": roc_auc_score(y_test, ann_prob)
}


print("\nANN Results:")
print(ann_results)


# =========================================================
# 11. 1D CNN MODEL
# =========================================================

X_train_cnn = X_train_dl.reshape(
    X_train_dl.shape[0],
    X_train_dl.shape[1],
    1
)

X_test_cnn = X_test_dl.reshape(
    X_test_dl.shape[0],
    X_test_dl.shape[1],
    1
)


cnn_model = Sequential([

    Conv1D(
        32,
        kernel_size=3,
        activation="relu",
        input_shape=(X_train_cnn.shape[1], 1)
    ),

    MaxPooling1D(pool_size=2),

    Conv1D(
        64,
        kernel_size=3,
        activation="relu"
    ),

    Flatten(),

    Dense(32, activation="relu"),

    Dropout(0.30),

    Dense(1, activation="sigmoid")
])


cnn_model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


print("\n===== TRAINING 1D CNN =====")

cnn_model.fit(
    X_train_cnn,
    y_train,
    validation_split=0.20,
    epochs=50,
    batch_size=32,
    verbose=1
)


cnn_prob = cnn_model.predict(X_test_cnn).ravel()

cnn_pred = (cnn_prob >= 0.5).astype(int)


cnn_results = {
    "Model": "1D CNN",
    "Accuracy": accuracy_score(y_test, cnn_pred),
    "Precision": precision_score(y_test, cnn_pred),
    "Recall": recall_score(y_test, cnn_pred),
    "F1 Score": f1_score(y_test, cnn_pred),
    "ROC-AUC": roc_auc_score(y_test, cnn_prob)
}


print("\n1D CNN Results:")
print(cnn_results)


# =========================================================
# 12. COMBINE ALL RESULTS
# =========================================================

dl_results_df = pd.DataFrame([
    ann_results,
    cnn_results
])


all_results = pd.concat(
    [
        ml_results_df,
        dl_results_df
    ],
    ignore_index=True
)


all_results = all_results.sort_values(
    by="F1 Score",
    ascending=False
)


print("\n====================================")
print("ALL MODEL RESULTS")
print("====================================")

print(all_results.to_string(index=False))


# Save results
all_results.to_csv(
    "outputs/model_comparison.csv",
    index=False
)


# =========================================================
# 13. MODEL COMPARISON GRAPH
# =========================================================

plt.figure(figsize=(14, 7))

all_results.set_index("Model")[
    [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC-AUC"
    ]
].plot(
    kind="bar",
    figsize=(14, 7)
)

plt.title(
    "Machine Learning and Deep Learning Model Comparison"
)

plt.ylabel("Score")

plt.ylim(0, 1)

plt.xticks(rotation=30)

plt.tight_layout()

plt.savefig(
    "outputs/model_comparison.png"
)

plt.close()


# =========================================================
# 14. SELECT BEST ML MODEL
# =========================================================

ml_results_df = ml_results_df.sort_values(
    by="F1 Score",
    ascending=False
)

best_model_name = ml_results_df.iloc[0]["Model"]

best_ml_model = ml_models[best_model_name]


print("\n====================================")
print("SELECTED BEST ML MODEL")
print("====================================")

print(best_model_name)


# =========================================================
# 15. SAVE BEST ML MODEL
# =========================================================

joblib.dump(
    best_ml_model,
    "models/titanic_best_model.pkl"
)

print("\nBest model saved successfully!")


# =========================================================
# 16. CONFUSION MATRIX
# =========================================================

best_ml_model.fit(
    X_train,
    y_train
)

best_pred = best_ml_model.predict(
    X_test
)


cm = confusion_matrix(
    y_test,
    best_pred
)


plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.title(
    f"Confusion Matrix - {best_model_name}"
)

plt.tight_layout()

plt.savefig(
    "outputs/confusion_matrix.png"
)

plt.close()


# =========================================================
# 17. CLASSIFICATION REPORT
# =========================================================

print("\n====================================")
print("CLASSIFICATION REPORT")
print("====================================")

print(
    classification_report(
        y_test,
        best_pred
    )
)


# =========================================================
# 18. FEATURE IMPORTANCE
# =========================================================

if best_model_name in [
    "Random Forest",
    "Gradient Boosting",
    "XGBoost"
]:

    model_object = best_ml_model.named_steps["model"]

    feature_names = (
        best_ml_model
        .named_steps["preprocessor"]
        .get_feature_names_out()
    )

    importances = model_object.feature_importances_

    feature_importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    })

    feature_importance_df = feature_importance_df.sort_values(
        by="Importance",
        ascending=False
    )

    print("\n===== FEATURE IMPORTANCE =====")

    print(
        feature_importance_df.head(15)
    )

    feature_importance_df.to_csv(
        "outputs/feature_importance.csv",
        index=False
    )


print("\n====================================")
print("PROJECT TRAINING COMPLETED")
print("====================================")