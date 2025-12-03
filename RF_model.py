import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, KFold, learning_curve
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

save_dir = r"D:\論文\資料預處理\RF"

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

# 建立 Random Forest 模型
rf_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=10,
    min_samples_split=8,
    min_samples_leaf=2,
    bootstrap=True,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)

# 預測
y_pred = rf_model.predict(X_test)
y_train_pred = rf_model.predict(X_train)

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
plt.title("Random Forest Predicted vs Actual")
plt.xlabel("Actual Values (Pa)")
plt.ylabel("Predicted Values (Pa)")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(save_dir, 'RF Predicted vs Actual.png'))
plt.show()

# 預測誤差分佈
errors = y_test - y_pred
plt.figure(figsize=(10, 6))
sns.histplot(errors, kde=True, bins=30, color='purple')
plt.title("RF Prediction Error Distribution")
plt.xlabel("Prediction Error (Pa)")
plt.ylabel("Frequency")
plt.grid(True)
plt.savefig(os.path.join(save_dir, 'Prediction Error Distribution.png'))
plt.show()

# 交叉驗證
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(rf_model, X_train, y_train, scoring='neg_mean_squared_error', cv=kf)
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
    rf_model, X_train, y_train, cv=kf, scoring='neg_mean_squared_error', 
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
plt.title('Learning Curve (Random Forest)')
plt.xlabel('Training Set Size')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(save_dir, 'Learning Curve (RF).png'))
plt.show()

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
predicted_pressure = rf_model.predict(custom_data)

print(f"自定義輸入的預測壓力值為: {predicted_pressure[0]:.2f} Pa")
