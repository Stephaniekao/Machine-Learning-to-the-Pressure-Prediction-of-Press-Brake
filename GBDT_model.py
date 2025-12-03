import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, KFold, learning_curve
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from datetime import datetime
from sklearn.model_selection import GridSearchCV

current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
save_dir = r"D:\論文\資料預處理\GBDT"

# Setting file paths
train_file = r"D:\論文\資料預處理\train.txt"
test_file = r"D:\論文\資料預處理\test.txt"

# Read training and testing datasets
train_data = pd.read_csv(train_file, sep='\t', encoding='utf-8')
test_data = pd.read_csv(test_file, sep='\t', encoding='utf-8')

# Separate X and y
X_train = train_data.drop(columns=['壓力(Pa)'])
y_train = train_data['壓力(Pa)']
X_test = test_data.drop(columns=['壓力(Pa)'])
y_test = test_data['壓力(Pa)']

# Load encoder and scaler
encoder = joblib.load(r"D:\python\encoder.pkl")
scaler = joblib.load(r"D:\python\scaler.pkl")
feature_names = encoder.get_feature_names_out(['材料'])

# Build GBDT model
#model = GradientBoostingRegressor(
#    n_estimators=150,            # Reduce number of trees
#    learning_rate=0.06,          # Lower learning rate
#    max_depth=3,                 # Reduce tree depth
#    min_samples_split=5,         # Lower split sensitivity
#    min_samples_leaf=5,          # Minimum samples per leaf
#    subsample=0.8,               # Add random sample subsampling
#    random_state=42
#)
#
#model.fit(X_train, y_train)
# --- 1) Build a "base model" ---
base_model = GradientBoostingRegressor(
    random_state=42
)

# --- 2) Set a "Quick parameter tuning grid" ---
param_grid = {
    "n_estimators":     [100, 150, 200, 250],
    "learning_rate":    [0.03, 0.05, 0.08, 0.1],
    "max_depth":        [2, 3, 4],
    "min_samples_split":[2, 5, 10],
    "min_samples_leaf": [3, 5, 7],
    "subsample":        [0.8, 1.0]
}

kf = KFold(n_splits=5, shuffle=True, random_state=42)

grid = GridSearchCV(
    estimator=base_model,
    param_grid=param_grid,
    scoring="neg_mean_squared_error",
    cv=kf,
    n_jobs=-1,
    verbose=1,
    return_train_score=True
)

# --- 3) Execute search to find the best GBDT ---
grid.fit(X_train, y_train)

print("Best params (GBDT):", grid.best_params_)
print("Best CV RMSE:", np.sqrt(-grid.best_score_))

# Use the model with the best parameters
model = grid.best_estimator_

# Predict
y_pred = model.predict(X_test)
y_train_pred = model.predict(X_train)

# Training and testing evaluation metrics
train_mse = mean_squared_error(y_train, y_train_pred)
train_rmse = np.sqrt(train_mse)
train_r2 = r2_score(y_train, y_train_pred)

test_mse = mean_squared_error(y_test, y_pred)
test_rmse = np.sqrt(test_mse)
test_r2 = r2_score(y_test, y_pred)

print(f"訓練集 MSE: {train_mse:.3f}, RMSE: {train_rmse:.3f}, R²: {train_r2:.3f}")
print(f"測試集 MSE: {test_mse:.3f}, RMSE: {test_rmse:.3f}, R²: {test_r2:.3f}")

# Visualize predicted vs actual values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', label='Predicted vs Actual')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.title("GBDT Predicted vs Actual")
plt.xlabel("Actual Values (Pa)")
plt.ylabel("Predicted Values (Pa)")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(save_dir, f'GBDT_Predicted_vs_Actual_{current_time}.png'))
plt.show()

# Prediction error distribution
errors = y_test - y_pred
plt.figure(figsize=(10, 6))
sns.histplot(errors, kde=True, bins=30, color='purple')
plt.title("GBDT Prediction Error Distribution")
plt.xlabel("Prediction Error (Pa)")
plt.ylabel("Frequency")
plt.grid(True)
plt.savefig(os.path.join(save_dir, f'Prediction_Error_Distribution_{current_time}.png'))
plt.show()

# Cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_train, y_train, scoring='neg_mean_squared_error', cv=kf)
cv_rmse = np.sqrt(-cv_scores)

print(f"RMSE for each fold: {cv_rmse}")
print(f"Average RMSE: {np.mean(cv_rmse):.3f}")

errors = y_test - y_pred
plt.figure(figsize=(10, 6))
plt.bar(range(1, len(cv_rmse) + 1), cv_rmse, color='blue')
plt.axhline(np.mean(cv_rmse), color='red', linestyle='--', label=f'Average RMSE: {np.mean(cv_rmse):.3f}')
plt.title("Cross-Validation Results")
plt.xlabel("Fold Number")
plt.ylabel("RMSE")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(save_dir, 'Cross_Validation_Results.png'))
plt.show()

# Learning curve
train_sizes, train_scores, test_scores = learning_curve(
    model, X_train, y_train, cv=kf, scoring='neg_mean_squared_error', 
    train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1
)

train_scores_mean = -np.mean(train_scores, axis=1)
test_scores_mean = -np.mean(test_scores, axis=1)
# Calculate the error difference  
error_difference = test_scores_mean - train_scores_mean
print("Error difference:", error_difference)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_scores_mean, label="Training Error", color='blue')
plt.plot(train_sizes, test_scores_mean, label="Validation Error", color='red')
plt.title('Learning Curve (GBDT)')
plt.xlabel('Training Set Size')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(save_dir, f'Learning_Curve_GBDT_{current_time}.png'))
plt.show()

np.save('GBDT_train_curve.npy', train_scores_mean)       # Each model saves its own
np.save('GBDT_val_curve.npy', test_scores_mean)

## Custom input test
custom_input = {
    '厚度(mm)': 1.5,
    '長度(mm)': 200,
    '角度(°)': 75,
    '材料': 'SPCC',
}

custom_input_df = pd.DataFrame(
    [[custom_input['材料']]],
    columns=['材料']  # The column name here should be consistent with the one used when training `encoder.pkl`
)
encoded_custom = encoder.transform(custom_input_df)
encoded_custom_df = pd.DataFrame(encoded_custom, columns=encoder.get_feature_names_out(['材料']))

numeric_custom = pd.DataFrame([[custom_input['厚度(mm)'], custom_input['長度(mm)'], custom_input['角度(°)']]],
                              columns=['厚度(mm)', '長度(mm)', '角度(°)'])
scaled_numeric_custom = scaler.transform(numeric_custom)
scaled_numeric_custom_df = pd.DataFrame(scaled_numeric_custom, columns=['厚度(mm)', '長度(mm)', '角度(°)'])

custom_data = pd.concat([scaled_numeric_custom_df, encoded_custom_df], axis=1)
predicted_pressure = model.predict(custom_data)

print(f"Predicted pressure for custom input: {predicted_pressure[0]:.2f} Pa")
