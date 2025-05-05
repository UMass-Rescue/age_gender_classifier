import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
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

# ---- Extra Features ----

# Maximum model confidence
df['max_confidence'] = df[['age_classify_v001_confidence', 'vit_age_classifier_confidence', 'fairface_classifier_confidence']].max(axis=1)

# Difference between max and min model confidence
df['confidence_diff_maxmin'] = df[['age_classify_v001_confidence', 'vit_age_classifier_confidence', 'fairface_classifier_confidence']].max(axis=1) - \
                               df[['age_classify_v001_confidence', 'vit_age_classifier_confidence', 'fairface_classifier_confidence']].min(axis=1)

# Agreement count between models
df['agreement_count'] = (
    (df['age_classify_v001_label'] == df['vit_age_classifier_label']).astype(int) +
    (df['age_classify_v001_label'] == df['fairface_classifier_label']).astype(int) +
    (df['vit_age_classifier_label'] == df['fairface_classifier_label']).astype(int)
)

# ---- Feature Set ----

X = df[
    [
        'age_classify_v001_label', 'age_classify_v001_confidence',
        'vit_age_classifier_label', 'vit_age_classifier_confidence',
        'fairface_classifier_label', 'fairface_classifier_confidence',
        'max_confidence',
        'confidence_diff_maxmin',
        'agreement_count'
    ]
]
y = df['true_range'].map(label_mapping)

# ---- Train/Test Split ----

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---- XGBoost Model ----

base_model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss')

# Expanded Hyperparameter Grid
param_grid = {
    'n_estimators': [100, 300, 500],
    'max_depth': [3, 5, 7, 9],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'gamma': [0, 1, 5]
}

# Grid Search
grid_search = GridSearchCV(
    estimator=base_model,
    param_grid=param_grid,
    cv=3,
    verbose=2,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

# Best model after search
best_model = grid_search.best_estimator_

# Predict and Evaluate
y_pred = best_model.predict(X_test)
print(classification_report(y_test, y_pred))
print("Best Hyperparameters:", grid_search.best_params_)
