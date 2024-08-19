import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import joblib

# Load your dataset
data = pd.read_csv('data/Connections.csv')

# Drop unnecessary columns
data = data.drop(['First_Name', 'Last_Name', 'URL', 'Email_Address'], axis=1)


# Convert 'DecisionMaker' to boolean if not already
data['DecisionMaker'] = data['DecisionMaker'].astype(bool)

# Convert 'KnownSince' to datetime and create additional features like year, month, etc.
data['KnownSince'] = pd.to_datetime(data['KnownSince'])
data['KnownSince_Year'] = data['KnownSince'].dt.year
data['KnownSince_Month'] = data['KnownSince'].dt.month

# Encode categorical variables
le = LabelEncoder()
data['Company'] = le.fit_transform(data['Company'])
data['Position'] = le.fit_transform(data['Position'])

# Define feature columns and target
X = data.drop(['Output_Target', 'KnownSince'], axis=1)  # Dropping 'KnownSince' as we've engineered new features
y = data['Output_Target'].map({'y': 1, 'n': 0})  # Encode target as 1/0

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define models
models = {
    'RandomForest': RandomForestClassifier(),
    'XGBoost': XGBClassifier()
}

# Train and evaluate models
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f'{name} Model')
    print('Accuracy:', accuracy_score(y_test, y_pred))
    print('Classification Report:\n', classification_report(y_test, y_pred))
    print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))
    print('-'*50)
    # Plotting Confusion Matrix for each model
    
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {name} Model')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    
# Hyperparameter tuning for the best model (example with RandomForest)
param_dist = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

random_search = RandomizedSearchCV(estimator=RandomForestClassifier(), param_distributions=param_dist, 
                                   n_iter=50, cv=5, n_jobs=-1, random_state=42, verbose=2)
random_search.fit(X_train, y_train)

best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test)
print('Best RandomForest Model after Tuning')
print('Accuracy:', accuracy_score(y_test, y_pred))
print('Classification Report:\n', classification_report(y_test, y_pred))
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))

# Feature Importance Analysis - Top 5 Features
importances = best_model.feature_importances_
indices = np.argsort(importances)[::-1][:5]  # Get top 5 feature indices

plt.figure(figsize=(10, 6))
plt.title("Top 5 Feature Importance")
sns.barplot(x=importances[indices], y=X.columns[indices])
plt.show()

# Plotting Model Performance - ROC Curve
plt.figure(figsize=(8, 6))
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc_score(y_test, y_pred_proba):.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend(loc='lower right')
plt.show()
# Compare models and decide the best model
best_model_name = ''
best_model_accuracy = 0

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    if accuracy > best_model_accuracy:
        best_model_accuracy = accuracy
        best_model_name = name

best_modelf = models[best_model_name]

# Evaluate the best model
best_modelf.fit(X_train, y_train)
y_pred = best_modelf.predict(X_test)
print('Best Model:', best_model_name)
print('Accuracy:', accuracy_score(y_test, y_pred))
print('Classification Report:\n', classification_report(y_test, y_pred))
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))

# Save the best model
joblib.dump(best_modelf, 'best_model.pkl')

# Next steps
# 1. Use the best model to make predictions on new data
# 2. Monitor the model's performance over time and retrain if necessary
# 3. Consider feature engineering and model tuning to further improve performance
# 4. Deploy the model in a production environment if it meets the desired performance criteria

