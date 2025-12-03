import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, KFold, learning_curve, GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from datetime import datetime
import json

current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
save_dir = r"D:\論文\資料預處理\SVR"

os.makedirs(save_dir, exist_ok=True)

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

# Build SVR model
#svr_model = SVR(
#    kernel='rbf',
#    C=1.5,                 # Slightly reduce penalty to avoid overfitting some folds
#    epsilon=0.08,          # Increase error tolerance to enhance stability
#    gamma='scale',
#    shrinking=True,
#    cache_size=500,
#    max_iter=10000,
#    tol=1e-4
#)

# --- 1) Define cross-validation ---
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# --- 2) Build base model (RBF kernel) ---
base_svr = SVR(kernel='rbf')

# --- 3) Parameter search space ---
# Version A: Quick grid (recommended to run this first)
param_grid_quick = {
    "C":       [0.5, 1, 2, 5, 10],
    "epsilon": [0.01, 0.05, 0.1, 0.2],
    "gamma":   ["scale", 0.01, 0.05, 0.1, 0.2]
}

# Version B: Wide grid (suitable for small datasets)
param_grid_wide = {
    "C":       np.logspace(-1, 2.5, 8),
    "epsilon": np.linspace(0.01, 0.3, 7),
    "gamma":   ["scale"] + list(np.logspace(-3, 0, 7))  # Key fix
}


# Choose which grid to use:
param_grid = param_grid_quick   # Use the quick one first; switch to param_grid_wide for finer search

# --- 4) GridSearchCV ---
grid = GridSearchCV(
    estimator=base_svr,
    param_grid=param_grid,
    scoring="neg_mean_squared_error",   # Use negative MSE as the evaluation metric
    cv=kf,
    n_jobs=-1,
    verbose=1,
    return_train_score=True
)

# --- 5) Execute search ---
grid.fit(X_train, y_train)

print("Best params:", grid.best_params_)
print("Best CV RMSE:", np.sqrt(-grid.best_score_))

# Extract the best model
svr_model = grid.best_estimator_

## (Optional) Save the best parameters and CV results
#with open(os.path.join(save_dir, f"svr_best_params_{current_time}.json"), "w", encoding="utf-8") as f:
#    json.dump({
#        "best_params": grid.best_params_,
#        "best_cv_rmse": float(np.sqrt(-grid.best_score_))
#    }, f, ensure_ascii=False, indent=2)
#
## (Optional) Save the model
#joblib.dump(svr_model, os.path.join(save_dir, f"svr_best_model_{current_time}.pkl"))

svr_model.fit(X_train, y_train)

# Predict
y_pred = svr_model.predict(X_test)
y_train_pred = svr_model.predict(X_train)

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
plt.title("SVR Predicted vs Actual")
plt.xlabel("Actual Values (Pa)")
plt.ylabel("Predicted Values (Pa)")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(save_dir, f'Prediction Error Distribution_{current_time}.png'))
#plt.show()

# Prediction error distribution
errors = y_test - y_pred
plt.figure(figsize=(10, 6))
sns.histplot(errors, kde=True, bins=30, color='purple')
plt.title("SVR Prediction Error Distribution")
plt.xlabel("Prediction Error (Pa)")
plt.ylabel("Frequency")
plt.grid(True)
plt.savefig(os.path.join(save_dir, f'Prediction Error Distribution_{current_time}.png'))
#plt.show()

# Cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(svr_model, X_train, y_train, scoring='neg_mean_squared_error', cv=kf)
cv_rmse = np.sqrt(-cv_scores)

print(f"RMSE for each fold: {cv_rmse}")
print(f"Average RMSE: {np.mean(cv_rmse):.3f}")

# Cross-validation results visualization
plt.figure(figsize=(10, 6))
plt.bar(range(1, len(cv_rmse) + 1), cv_rmse, color='blue')
plt.axhline(np.mean(cv_rmse), color='red', linestyle='--', label=f'Average RMSE: {np.mean(cv_rmse):.3f}')
plt.title("Cross-Validation Results")
plt.xlabel("Fold Number")
plt.ylabel("RMSE")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(save_dir, 'Cross-Validation Results_{current_time}'))
#plt.show()

# Learning curve
train_sizes, train_scores, test_scores = learning_curve(
    svr_model, X_train, y_train, cv=kf, scoring='neg_mean_squared_error', 
    train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1
)

train_scores_mean = -np.mean(train_scores, axis=1)
test_scores_mean = -np.mean(test_scores, axis=1)
# Calculate the error difference for each training set size
error_difference = test_scores_mean - train_scores_mean
print("Error difference:", error_difference)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_scores_mean, label="Training Error", color='blue')
plt.plot(train_sizes, test_scores_mean, label="Validation Error", color='red')
plt.title('Learning Curve (SVR)')
plt.xlabel('Training Set Size')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(save_dir, 'Learning Curve (SVR)_{current_time}'))
#plt.show()

np.save('train_sizes.npy', train_sizes)         # Save only once
np.save('svr_train_curve.npy', train_scores_mean)       # Each model saves its own
np.save('svr_val_curve.npy', test_scores_mean)

## Custom input test
custom_input = {
    '厚度(mm)': 1.5,
    '長度(mm)': 200,
    '角度(°)': 75,
    '材料': 'SPCC'
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
predicted_pressure = svr_model.predict(custom_data)

print(f"Predicted pressure for custom input: {predicted_pressure[0]:.2f} Pa")

# Default parameters for generating all combinations
materials = ['SPCC', 'SUS', 'SUS_Tape']
lengths = [50, 100, 150, 200, 250]
thicknesses = [1.0, 1.5, 2.0]
angles = np.arange(0, 91, 1)  # 0° ~ 90°

# Tool and die correspondence logic
def get_tool_and_die(thickness):
    if thickness == 1.0:
        return '尖刀', '6V'
    elif thickness == 1.5:
        return '尖刀', '8V'
    elif thickness == 2.0:
        return '劍刀', '10V'
    else:
        raise ValueError(f"Unknown thickness: {thickness}")

# Collect all data
records = []

for mat in materials:
    for t in thicknesses:
        for l in lengths:
            tool, die = get_tool_and_die(t)
            for ang in angles:
                records.append({
                    '材料': mat,
                    '厚度(mm)': t,
                    '長度(mm)': l,
                    '角度(°)': ang,
                    '刀具': tool,
                    '下模': die
                })

df_all = pd.DataFrame(records)

# Categorical feature encoding
category_cols = ['材料']
encoded = encoder.transform(df_all[category_cols])
encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(category_cols))

# Numerical feature scaling
numeric_cols = ['厚度(mm)', '長度(mm)', '角度(°)']
scaled = scaler.transform(df_all[numeric_cols])
scaled_df = pd.DataFrame(scaled, columns=numeric_cols)

# Combine as model input
model_input = pd.concat([scaled_df, encoded_df], axis=1)

# Predict pressure for all combinations
df_all['Predicted Pressure (Pa)'] = svr_model.predict(model_input)

plt.figure(figsize=(14, 8))

# Plot curves for each group (Material + Thickness + Length)
for (mat, t, l), group in df_all.groupby(['材料', '厚度(mm)', '長度(mm)']):
    label = f'{mat} / {t}mm / {l}mm'
    plt.plot(group['角度(°)'], group['Predicted Pressure (Pa)'], label=label)

plt.xlabel('Angle (°)', fontsize=14, fontweight='bold')
plt.ylabel('Predicted Pressure (Pa)', fontsize=14, fontweight='bold')
# Set axis ticks
plt.xticks(np.arange(0, 91, 2))  # X-axis from 0 to 90, every 2 degrees
plt.xlim(0, 90)
plt.yticks(np.arange(0, df_all['Predicted Pressure (Pa)'].max() + 0.5, 0.5))  # Y-axis from 0, every 0.5 units
plt.title('Angle vs Predicted Pressure (All Combinations)', fontsize=16, fontweight='bold')
plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=9)
plt.tight_layout(rect=[0, 0, 0.85, 1])  # Reserve space on the right
plt.grid(True)
plt.tight_layout()
plt.show()

df_all.to_excel(r'D:\論文\角度對壓力_預測結果.xlsx', index=False)
