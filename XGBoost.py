import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, KFold, learning_curve, RandomizedSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from datetime import datetime
import json

current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
save_dir = r"D:\論文\資料預處理\XGBoost"

# 設定檔案路徑
train_file = r"D:\論文\資料預處理\train.txt"
test_file = r"D:\論文\資料預處理\test.txt"

# 讀取訓練集與測試集
train_data = pd.read_csv(train_file, sep='\t', encoding='utf-8')
test_data = pd.read_csv(test_file, sep='\t', encoding='utf-8')

# 分離 X 與 y
X_train = train_data.drop(columns=['壓力(Pa)'])
y_train = train_data['壓力(Pa)']
X_test = test_data.drop(columns=['壓力(Pa)'])
y_test = test_data['壓力(Pa)']

# 加載編碼器與標準化器
encoder = joblib.load(r"D:\python\encoder.pkl")
scaler = joblib.load(r"D:\python\scaler.pkl")
feature_names = encoder.get_feature_names_out(['材料', '刀具', '下模'])

# 建立 XGBoost 模型
xgb_model = XGBRegressor(
    n_estimators=250,
    learning_rate=0.06,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.5,
    reg_lambda=1,
    random_state=42
)

#base_xgb = XGBRegressor(
#    objective="reg:squarederror",
#    n_estimators=400,         # 搭配 early_stopping，用大一點沒關係
#    learning_rate=0.08,
#    max_depth=4,
#    subsample=0.9,
#    colsample_bytree=0.9,
#    reg_alpha=0.5,
#    reg_lambda=1.0,
#    min_child_weight=1,
#    random_state=42,
#    n_jobs=-1,
#    tree_method="hist"        # 一般 CPU 訓練用這個會比較快
#)

# -----------------------------
# 2) 建立超參數搜尋空間（Quick 版）
# -----------------------------
base_xgb = XGBRegressor(
    objective="reg:squarederror",
    n_estimators=400,         # 搭配 early_stopping，用大一點沒關係
    learning_rate=0.08,
    max_depth=4,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_alpha=0.5,
    reg_lambda=1.0,
    min_child_weight=1,
    random_state=42,
    n_jobs=-1,
    tree_method="hist"        # 一般 CPU 訓練用這個會比較快
)

param_dist = {
    "n_estimators":   [150, 200, 250, 300, 400],
    "max_depth":      [3, 4, 5, 6],
    "learning_rate":  [0.03, 0.05, 0.08, 0.1, 0.15],
    "subsample":      [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    "min_child_weight": [1, 2, 3, 4],
    "reg_alpha":      [0, 0.1, 0.5, 1.0],
    "reg_lambda":     [0.5, 1.0, 2.0]
}

kf = KFold(n_splits=5, shuffle=True, random_state=42)

random_search = RandomizedSearchCV(
    estimator=base_xgb,
    param_distributions=param_dist,
    n_iter=40,                          # 搜 40 組就很夠（資料不大）
    scoring="neg_mean_squared_error",
    cv=kf,
    n_jobs=-1,
    verbose=1,
    random_state=42
)
random_search.fit(X_train, y_train)

best_params = random_search.best_params_
best_rmse = np.sqrt(-random_search.best_score_)
print("Best params (XGBoost):", best_params)
print(f"Best CV RMSE: {best_rmse:.3f}")

# 取出最佳模型
xgb_model = random_search.best_estimator_

xgb_model.set_params(n_estimators=500)   # 給它更大上限
xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    eval_metric="rmse",
    verbose=False,
    early_stopping_rounds=30
)
print("Best n_estimators after early_stopping:", xgb_model.best_iteration)

# 預測
y_pred = xgb_model.predict(X_test)
y_train_pred = xgb_model.predict(X_train)

# 訓練與測試評估指標
train_mse = mean_squared_error(y_train, y_train_pred)
train_rmse = np.sqrt(train_mse)
train_r2 = r2_score(y_train, y_train_pred)

test_mse = mean_squared_error(y_test, y_pred)
test_rmse = np.sqrt(test_mse)
test_r2 = r2_score(y_test, y_pred)

print(f"訓練集 MSE: {train_mse:.3f}, RMSE: {train_rmse:.3f}, R²: {train_r2:.3f}")
print(f"測試集 MSE: {test_mse:.3f}, RMSE: {test_rmse:.3f}, R²: {test_r2:.3f}")

# 可視化預測與實際值
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', label='Predicted vs Actual')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.title("XGBoost Predicted vs Actual")
plt.xlabel("Actual Values (Pa)")
plt.ylabel("Predicted Values (Pa)")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(save_dir, f'Prediction Error Distribution_{current_time}.png'))
plt.show()

# 預測誤差分佈
errors = y_test - y_pred
plt.figure(figsize=(10, 6))
sns.histplot(errors, kde=True, bins=30, color='purple')
plt.title("XGBoost Prediction Error Distribution")
plt.xlabel("Prediction Error (Pa)")
plt.ylabel("Frequency")
plt.grid(True)
plt.savefig(os.path.join(save_dir, f'Prediction Error Distribution_{current_time}.png'))
plt.show()

# 交叉驗證
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(xgb_model, X_train, y_train, scoring='neg_mean_squared_error', cv=kf)
cv_rmse = np.sqrt(-cv_scores)

print(f"每折的 RMSE: {cv_rmse}")
print(f"平均 RMSE: {np.mean(cv_rmse):.3f}")

# 交叉驗證結果視覺化
plt.figure(figsize=(10, 6))
plt.bar(range(1, len(cv_rmse) + 1), cv_rmse, color='blue')
plt.axhline(np.mean(cv_rmse), color='red', linestyle='--', label=f'Average RMSE: {np.mean(cv_rmse):.3f}')
plt.title("Cross-Validation Results")
plt.xlabel("Fold Number")
plt.ylabel("RMSE")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(save_dir, 'Cross-Validation Results.png'))
plt.show()

# 學習曲線
train_sizes, train_scores, test_scores = learning_curve(
    xgb_model, X_train, y_train, cv=kf, scoring='neg_mean_squared_error', 
    train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1
)

train_scores_mean = -np.mean(train_scores, axis=1)
test_scores_mean = -np.mean(test_scores, axis=1)
# 計算誤差差距
error_difference = test_scores_mean - train_scores_mean
print("誤差差距:", error_difference)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_scores_mean, label="Training Error", color='blue')
plt.plot(train_sizes, test_scores_mean, label="Validation Error", color='red')
plt.title('Learning Curve (XGBoost)')
plt.xlabel('Training Set Size')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(save_dir, f'Learning Curve (XGBoost)_{current_time}.png'))
plt.show()

np.save('XGBoost_train_curve.npy', train_scores_mean)       # 每個模型各自存一個
np.save('XGBoost_val_curve.npy', test_scores_mean)

# 自定義輸入測試
custom_input = {
    '厚度(mm)': 1.5,
    '長度(mm)': 100,
    '角度(°)': 30,
    '材料': 'SPCC',
    '刀具': '尖刀',
    '下模': '8V',
}

# 處理自定義輸入
custom_input_df = pd.DataFrame(
    [[custom_input['材料'], custom_input['刀具'], custom_input['下模']]],
    columns=['材料', '刀具', '下模']  # 這裡的欄位名稱要與訓練 `encoder.pkl` 時保持一致
)
encoded_custom = encoder.transform(custom_input_df)
encoded_custom_df = pd.DataFrame(encoded_custom, columns=encoder.get_feature_names_out(['材料', '刀具', '下模']))

numeric_custom = pd.DataFrame([[custom_input['厚度(mm)'], custom_input['長度(mm)'], custom_input['角度(°)']]],
                              columns=['厚度(mm)', '長度(mm)', '角度(°)'])
scaled_numeric_custom = scaler.transform(numeric_custom)
scaled_numeric_custom_df = pd.DataFrame(scaled_numeric_custom, columns=['厚度(mm)', '長度(mm)', '角度(°)'])

custom_data = pd.concat([scaled_numeric_custom_df, encoded_custom_df], axis=1)
predicted_pressure = xgb_model.predict(custom_data)
print(f"自定義輸入的預測壓力值為: {predicted_pressure[0]:.2f} Pa")
