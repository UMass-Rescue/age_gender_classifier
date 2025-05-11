import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier

# Load your CSV
df = pd.read_csv('pivoted_output.csv')  

# Convert true_label into an age range
def age_to_range(age):
    if age <= 2:
        return "0-2"
    elif age <= 9:
        return "3-9"
    elif age <= 19:
        return "10-19"
    elif age <= 29:
        return "20-29"
    elif age <= 39:
        return "30-39"
    elif age <= 49:
        return "40-49"
    elif age <= 59:
        return "50-59"
    elif age <= 69:
        return "60-69"
    else:
        return "more than 70"

df['true_range'] = df['true_label'].apply(age_to_range)

# Fix inconsistent model output labels
def fix_label(label):
    if label == "70-79":
        return "more than 70"
    return label

df['age_classify_v001_label'] = df['age_classify_v001_label'].apply(fix_label)
df['vit_age_classifier_label'] = df['vit_age_classifier_label'].apply(fix_label)
df['fairface_classifier_label'] = df['fairface_classifier_label'].apply(fix_label)

# Map all labels to integers
label_mapping = {
    "0-2": 0,
    "3-9": 1,
    "10-19": 2,
    "20-29": 3,
    "30-39": 4,
    "40-49": 5,
    "50-59": 6,
    "60-69": 7,
    "more than 70": 8
}
df['age_classify_v001_label'] = df['age_classify_v001_label'].map(label_mapping)
df['vit_age_classifier_label'] = df['vit_age_classifier_label'].map(label_mapping)
df['fairface_classifier_label'] = df['fairface_classifier_label'].map(label_mapping)

# Feature and target setup
X = df[
    [
        'age_classify_v001_label', 'age_classify_v001_confidence',
        'vit_age_classifier_label', 'vit_age_classifier_confidence',
        'fairface_classifier_label', 'fairface_classifier_confidence'
    ]
]
y = df['true_range'].map(label_mapping)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Final XGBoost model using the best hyperparameters
clf = XGBClassifier(
    random_state=42,
    use_label_encoder=False,
    eval_metric='mlogloss',
    colsample_bytree=0.6,
    gamma=0,
    learning_rate=0.01,
    max_depth=3,
    n_estimators=500,
    subsample=0.6
)

# Train
clf.fit(X_train, y_train)

# Predict and evaluate
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

# ---- Plot Confusion Matrix ----

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Define readable class labels
class_labels = [
    "0-2", "3-9", "10-19", "20-29", "30-39", 
    "40-49", "50-59", "60-69", "70+"
]

# Plot
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_labels, yticklabels=class_labels)

plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
