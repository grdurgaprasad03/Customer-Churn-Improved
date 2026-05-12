import pandas as pd
import pickle

# =========================
# LOAD DATA
# =========================

print("Loading dataset...")

df = pd.read_csv('churn.csv')

# =========================
# CLEAN COLUMN NAMES
# =========================

df.columns = df.columns.str.strip()

# =========================
# DROP CUSTOMER ID
# =========================

if 'customerID' in df.columns:
    df.drop('customerID', axis=1, inplace=True)

# =========================
# HANDLE MISSING VALUES
# =========================

if 'TotalCharges' in df.columns:
    df['TotalCharges'] = pd.to_numeric(
        df['TotalCharges'],
        errors='coerce'
    )

df.dropna(inplace=True)

# =========================
# CONVERT CATEGORICAL DATA
# =========================

print("Encoding categorical variables...")

df = pd.get_dummies(df, drop_first=True)

# =========================
# SPLIT FEATURES & TARGET
# =========================

X = df.drop('Churn_Yes', axis=1)
y = df['Churn_Yes']

# Save feature names
feature_columns = X.columns

# =========================
# TRAIN TEST SPLIT
# =========================

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================
# FEATURE SCALING
# =========================

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =========================
# MODEL TRAINING
# =========================

print("Training XGBoost model...")

from xgboost import XGBClassifier

model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='logloss'
)

model.fit(X_train, y_train)

# =========================
# MODEL EVALUATION
# =========================

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)

predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)
auc = roc_auc_score(y_test, probabilities)

print("\n================ MODEL PERFORMANCE ================")
print(f"Accuracy      : {accuracy:.4f}")
print(f"Precision     : {precision:.4f}")
print(f"Recall        : {recall:.4f}")
print(f"F1 Score      : {f1:.4f}")
print(f"ROC-AUC Score : {auc:.4f}")

print("\n================ CONFUSION MATRIX ================")
print(confusion_matrix(y_test, predictions))

print("\n================ CLASSIFICATION REPORT ================")
print(classification_report(y_test, predictions))

# =========================
# FEATURE IMPORTANCE
# =========================

importance_df = pd.DataFrame({
    'Feature': feature_columns,
    'Importance': model.feature_importances_
})

importance_df = importance_df.sort_values(
    by='Importance',
    ascending=False
)

print("\n================ TOP 10 IMPORTANT FEATURES ================")
print(importance_df.head(10))

# =========================
# SAVE FILES
# =========================

pickle.dump(model, open('model.pkl', 'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))
pickle.dump(feature_columns, open('features.pkl', 'wb'))
pickle.dump(importance_df, open('importance.pkl', 'wb'))
pickle.dump(accuracy, open('accuracy.pkl', 'wb'))

print("\n✅ Training complete and files saved successfully!")