import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score

# 👉 Step 1: Load dataset
df = pd.read_csv("E:/Healthcare_Document_classification/dataset.csv")

# ✅ Step 2: Shuffle dataset to avoid order bias
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# 👉 Step 3: Extract features and labels
X = df["medical_abstract"]
y = df["condition_label"]

# 👉 Step 4: Vectorize the text
vectorizer = TfidfVectorizer(max_features=5000)
X_vectorized = vectorizer.fit_transform(X)

# 👉 Step 5: Split the data
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

# 👉 Step 6: Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 👉 Step 7: Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy: {accuracy:.2%}")

# 📊 Extra: Detailed classification report
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

# 🧩 Extra: Confusion matrix
print("🧩 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# 📉 Extra: Cross-validation
cv_scores = cross_val_score(model, X_vectorized, y, cv=5)
print(f"\n📉 Cross-Validation Accuracy: {cv_scores.mean():.2%} ± {cv_scores.std():.2%}")

# 👉 Step 8: Save model and vectorizer
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)
