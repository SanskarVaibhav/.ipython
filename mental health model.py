import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import shap

# Step 1: Load the dataset
df = pd.read_csv(r"C:\Users\Sanskar Vaibhav\Downloads\mental_health_analysis.csv")  # Fixed file path issue
print("Dataset Loaded Successfully!\n")
print(df.head())  # Display first 5 rows

# Step 2: Data Preprocessing
# Check for missing values
print("Missing Values in Dataset:\n", df.isnull().sum())
df = df.dropna()  # Drop rows with missing values

# Convert categorical columns to numerical (if any)
label_encoders = {}
for column in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le  # Store encoder for later use

# Step 3: Feature Engineering (Creating new meaningful features)
if 'screen_time_per_day' in df.columns and 'social_media_usage' in df.columns:
    df['screen_social_interaction'] = df['screen_time_per_day'] * df['social_media_usage']
else:
    print("Warning: 'screen_time_per_day' or 'social_media_usage' not found. Skipping feature creation.")

if 'stress_level' in df.columns and 'physical_activity' in df.columns:
    df['stress_physical_effect'] = df['stress_level'] * df['physical_activity']
else:
    print("Warning: 'stress_level' or 'physical_activity' not found. Skipping feature creation.")

if 'sleep_hours' in df.columns and 'caffeine_intake' in df.columns:
    df['sleep_caffeine_ratio'] = df['sleep_hours'] / (df['caffeine_intake'] + 1)  # Avoid division by zero
else:
    print("Warning: 'sleep_hours' or 'caffeine_intake' not found. Skipping feature creation.")

# Step 4: Exploratory Data Analysis (EDA)
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()

# Step 5: Feature Selection (Assume 'mental_health_condition' is the target column)
if 'mental_health_condition' not in df.columns:
    print("Error: 'mental_health_condition' column not found in dataset.")
    print("Available columns:", df.columns)
    exit()

X = df.drop(columns=['mental_health_condition'])  # Features
y = df['mental_health_condition']  # Target

# Normalize numerical features
scaler = StandardScaler()
numeric_columns = X.select_dtypes(include=['int64', 'float64']).columns
X[numeric_columns] = scaler.fit_transform(X[numeric_columns])

# Feature Selection using Feature Importance
model_temp = RandomForestClassifier(n_estimators=100, random_state=42)
model_temp.fit(X, y)
feature_importances = pd.Series(model_temp.feature_importances_, index=X.columns)
selected_features = feature_importances[feature_importances > 0.01].index  # Keep only important features
X = X[selected_features]
print("Selected Features:", selected_features)

# Compute P-values using Logistic Regression
X_const = sm.add_constant(X)  # Add constant term for intercept
logit_model = sm.Logit(y, X_const)
result = logit_model.fit()
print(result.summary())  # Displays p-values for each feature

# Step 6: Handle Imbalanced Data using SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X, y)

# Step 7: Split Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X_train_resampled, y_train_resampled, test_size=0.2, random_state=42)
print("Data Split into Training and Testing Sets.\n")

# Step 8: Train the Machine Learning Model
model = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=5, min_samples_leaf=2, random_state=42)
model.fit(X_train, y_train)
print("Model Trained Successfully!\n")

# Step 9: Model Evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Step 10: Optimization (Hyperparameter Tuning with GridSearchCV)
param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [10, 20, None], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]}
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)
print("Best Parameters:", grid_search.best_params_)

# Train the final model with best parameters
best_model = RandomForestClassifier(**grid_search.best_params_, random_state=42)
best_model.fit(X_train, y_train)
y_best_pred = best_model.predict(X_test)
best_accuracy = accuracy_score(y_test, y_best_pred)
print("Optimized Model Accuracy:", best_accuracy)

# Step 11: Test XGBoost Model for Comparison
xgb_model = XGBClassifier(n_estimators=100, learning_rate=0.05)
xgb_model.fit(X_train, y_train)
y_xgb_pred = xgb_model.predict(X_test)
xgb_accuracy = accuracy_score(y_test, y_xgb_pred)
print("XGBoost Model Accuracy:", xgb_accuracy)

# Step 11.5: Test Linear Regression Model for Comparison
lr_model = LogisticRegression(max_iter=1000)  # Using Logistic Regression since this is a classification task
lr_model.fit(X_train, y_train)
y_lr_pred = lr_model.predict(X_test)
lr_accuracy = accuracy_score(y_test, y_lr_pred)
print("\nLinear Model Performance:")
print("Accuracy:", lr_accuracy)
print("Classification Report:\n", classification_report(y_test, y_lr_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_lr_pred))

# Calculate and display feature coefficients
feature_coef = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': lr_model.coef_[0]
})
print("\nLinear Model Feature Coefficients:")
print(feature_coef.sort_values(by='Coefficient', ascending=False))

# Step 12: SHAP Analysis for Model Interpretability
explainer = shap.Explainer(best_model)
shap_values = explainer(X_test)
shap.summary_plot(shap_values, X_test)
