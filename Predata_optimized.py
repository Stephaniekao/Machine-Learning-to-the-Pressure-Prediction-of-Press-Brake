import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
#from sklearn.feature_selection import SelectFromModel
from scipy.stats import jarque_bera
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# 1. Read data
data = pd.read_excel(r"D:\論文\論文數據整理.xlsx")

numeric_features = ['厚度(mm)', '長度(mm)', '角度(°)']  # Numeric features
categorical_features = ['材料']  # Categorical features
target = '壓力(Pa)'  # Target column

# Split dataset
X = data[numeric_features + categorical_features].copy()
y = data[target].copy()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 2. Display data structure
print("Data Info")
print(data.info())
#print(data)

print("Data preprocessing")
# Convert non-numeric features to numeric before data cleaning
# One-hot encode categorical features

# 3. Data cleaning
print("Data cleaning")
# Handle missing values (interpolation)
print("Handling missing values")
# Check missing values before interpolation
print("Missing values before interpolation:")
print(data.isnull().sum())

# ====== Minimal necessary adjustments for "split then process" ======
# Only handle missing values in the training set: numeric with linear interpolation + fallback median; categorical with mode
# (Test set always uses "rules learned from training set" to transform)

# Numeric: interpolate training set (by column), then fill remaining missing with median; test set fills with training median
X_train_num = X_train[numeric_features].copy()
X_test_num  = X_test[numeric_features].copy()

# Interpolate training set by column
for c in numeric_features:
    X_train_num[c] = X_train_num[c].interpolate(method='linear', axis=0)

# If still missing values, fill with training median
train_medians = X_train_num.median()
X_train_num = X_train_num.fillna(train_medians)
# Test set missing values filled with training median
X_test_num = X_test_num.fillna(train_medians)

# Categorical: fill training set with mode; test set with training mode
X_train_cat = X_train[categorical_features].copy()
X_test_cat  = X_test[categorical_features].copy()
train_modes = X_train_cat.mode(dropna=True).iloc[0]
X_train_cat = X_train_cat.fillna(train_modes)
X_test_cat  = X_test_cat.fillna(train_modes)

# Check missing values after interpolation (using your original print)
print("Missing values after interpolation:")
tmp_check = pd.concat([X_train_num, X_train_cat], axis=1)
print(tmp_check.isnull().sum())

# Outlier handling (using IQR range to detect outliers)
print(len(data))
print("Outlier handling")

# Only use "training set" to estimate IQR bounds; clip both training and test sets to bounds, do not remove rows (more stable for small samples)
def compute_iqr_bounds(s: pd.Series):
    Q1 = s.quantile(0.25)
    Q3 = s.quantile(0.75)
    IQR = Q3 - Q1
    return Q1 - 1.5 * IQR, Q3 + 1.5 * IQR

iqr_bounds_dict = {}
for col in numeric_features:
    lower_bound, upper_bound = compute_iqr_bounds(X_train_num[col])
    iqr_bounds_dict[col] = (lower_bound, upper_bound)
    # Print the bounds
    print(lower_bound, upper_bound)

    # Use clip (do not remove rows)
    before_outliers_train = ((X_train_num[col] < lower_bound) | (X_train_num[col] > upper_bound)).sum()
    before_outliers_test  = ((X_test_num[col]  < lower_bound) | (X_test_num[col]  > upper_bound)).sum()

    X_train_num[col] = X_train_num[col].clip(lower_bound, upper_bound)
    X_test_num[col]  = X_test_num[col].clip(lower_bound, upper_bound)

    # Keep your original message, but add explanation that clip is used this time:
    if (before_outliers_train + before_outliers_test) > 0:
        print("Outliers have been handled (using clip to boundary, no rows removed)")
    else:
        print("No outliers detected")

# Check and remove duplicates
print("Removing duplicates")
print(len(data))
data = data.drop_duplicates()
print(len(data))
print(data)

print("獨熱編碼")
materials = ['SPCC', 'SUS', 'SUS_Tape']
tools = ['劍刀', '尖刀']
molds = ['10V', '8V', '6V']

# Create OneHotEncoder, explicitly specify category order
# >>> Minimal fix: do not hardcode categories, add handle_unknown='ignore', but keep your variable names encoder/encoded_df
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

# Fit only on training set
encoded_features_train = encoder.fit_transform(X_train_cat[categorical_features])
encoded_features_test  = encoder.transform(X_test_cat[categorical_features])

# Additionally: if you still want to create an encoded version of the "full data" (just for description/plotting),
# you must use the same encoder to transform (do not refit):
encoded_features_full  = encoder.transform(data[categorical_features].fillna(train_modes))

print(data[categorical_features])
encoded_df = pd.DataFrame(encoded_features_full, columns=encoder.get_feature_names_out(categorical_features))
data = pd.concat([data.drop(columns=categorical_features).reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)

# Save the result to a file
data.to_csv(r"D:\論文\資料預處理\encoded_features_cleaned.txt", 
            sep='\t', index=False, encoding='utf-8')

# 4. Feature processing
print("Feature processing")
data_origin = data[['厚度(mm)', '長度(mm)', '角度(°)', '壓力(Pa)']]

# Standardization
scaler_standard = StandardScaler()

# >>> Minimal fix: scaler only fit on "training set numerical features", then transform others
data_standard_train = scaler_standard.fit_transform(X_train_num[['厚度(mm)', '長度(mm)', '角度(°)']])
data_standard_test  = scaler_standard.transform(X_test_num[['厚度(mm)', '長度(mm)', '角度(°)']])
# If you still need a standardized version for the "full data" (for visualization/output), only do transform (do not fit again)
data_standard_full  = scaler_standard.transform(data[['厚度(mm)', '長度(mm)', '角度(°)']])

# Keep your original variable name: data_standard (using the full version here to avoid changing your subsequent code)
data_standard = pd.DataFrame(data_standard_full, columns=['厚度(mm)', '長度(mm)', '角度(°)'])
print(data_standard.dtypes)

# Dictionary mapping Chinese column names to English column names
column_names = {
    '厚度(mm)': 'Thickness (mm)',
    '長度(mm)': 'Length',
    '角度(°)': 'Angle (°)',
    '壓力(Pa)': 'Pressure (Pa)'
}

for col in ['厚度(mm)', '長度(mm)', '角度(°)']:
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))

    # Upper plot: Original data KDE
    sns.kdeplot(data_origin[col], ax=axes[0], color='blue', label='Original Data', linewidth=2)
    axes[0].set_title(f'{column_names[col]} - Original Distribution', fontsize=16, fontweight='bold')
    axes[0].set_xlabel(column_names[col], fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Probability Density', fontsize=14, fontweight='bold')
    axes[0].tick_params(axis='both', labelsize=12)
    axes[0].legend(fontsize=12)
    axes[0].grid(True)

    # Lower plot: Standardized data KDE (using the same scaler's transform result)
    sns.kdeplot(data_standard[col], ax=axes[1], color='red', label='Z-Score Standardized', linewidth=2)
    axes[1].set_title(f'{column_names[col]} - Z-Score Standardized Distribution', fontsize=16, fontweight='bold')
    axes[1].set_xlabel(column_names[col], fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Probability Density', fontsize=14, fontweight='bold')
    axes[1].tick_params(axis='both', labelsize=12)
    axes[1].legend(fontsize=12)
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()

data_standard.to_csv(r"D:\論文\資料預處理\data_standard_324.txt", 
                     sep='\t',   # Use tab as the delimiter
                     index=True,  # Do not save the index column
                     header=True,  # Keep the header row
                     encoding='utf-8')  # Ensure no garbled characters for Chinese

data_standard_df = pd.DataFrame(data_standard, columns=['厚度(mm)', '長度(mm)', '角度(°)'])
encoded_df = data[['材料_SPCC', '材料_SUS','材料_SUS_Tape']]
encoded_df.to_csv(r"D:\論文\資料預處理\data_standard_3241.txt", 
                     sep='\t',   # Use tab as the delimiter
                     index=True,  # Do not save the index column
                     header=True,  # Keep the header row
                     encoding='utf-8')  # Ensure no garbled characters for Chinese
merged_data = pd.concat([data_standard_df, encoded_df, data['壓力(Pa)']], axis=1)  

# 4. Save as txt file
merged_data.to_csv(r"D:\論文\資料預處理\data_standard_3242.txt", 
                   sep='\t',   # Use tab as the delimiter
                   index=True,  # Do not save the index column
                   header=True,  # Keep the header row
                   encoding='utf-8')  # Ensure no garbled characters for Chinese

joblib.dump(encoder, r"D:\python\encoder.pkl")
joblib.dump(scaler_standard, r"D:\python\scaler.pkl")

# Use the same set of "processed" features to output train/test (keep your original variable names)
# Here, the final features are "training/testing numerical (standardized) + training/testing categorical (OneHot encoded)"
X_train_final = pd.concat([
    pd.DataFrame(scaler_standard.transform(X_train_num), columns=numeric_features, index=X_train.index),
    pd.DataFrame(encoder.transform(X_train_cat), columns=encoder.get_feature_names_out(categorical_features), index=X_train.index)
], axis=1)

X_test_final = pd.concat([
    pd.DataFrame(scaler_standard.transform(X_test_num), columns=numeric_features, index=X_test.index),
    pd.DataFrame(encoder.transform(X_test_cat), columns=encoder.get_feature_names_out(categorical_features), index=X_test.index)
], axis=1)

# Merge X_train and y_train
train_data = pd.concat([X_train_final, y_train], axis=1)

# Merge X_test and y_test
test_data = pd.concat([X_test_final, y_test], axis=1)

# Save merged Train data
train_data.to_csv(r"D:\論文\資料預處理\train.txt", 
                   sep='\t',   # Use tab as the delimiter
                   index=False,  # Do not save the index column
                   header=True,  # Keep the header row
                   encoding='utf-8')  # Ensure no garbled characters for Chinese

test_data.to_csv(r"D:\論文\資料預處理\test.txt", 
                   sep='\t',   # Use tab as the delimiter
                   index=False,  # Do not save the index column
                   header=True,  # Keep the header row
                   encoding='utf-8')  # Ensure no garbled characters for Chinese

# Visualize dataset shapes
for col in ['厚度(mm)', '長度(mm)', '角度(°)']:
    plt.figure(figsize=(12, 6))
    
    # KDE of original standardized data (using the same scaler's transform result)
    sns.kdeplot(data_standard[col], color='blue', label='Original Data', linewidth=2)
    # KDE of training data (standardized)
    sns.kdeplot(X_train_final[col], color='green', label='Training Data', linewidth=2, linestyle='--')
    # KDE of testing data (standardized)
    sns.kdeplot(X_test_final[col], color='red', label='Testing Data', linewidth=2, linestyle=':')

    # Title and labels (larger, bold)
    plt.title(f'{column_names[col]} - Distribution Before and After Data Splitting',
              fontsize=16, fontweight='bold')
    plt.xlabel(column_names[col], fontsize=14, fontweight='bold')
    plt.ylabel('Probability Density', fontsize=14, fontweight='bold')

    # Tick label size
    plt.tick_params(axis='both', labelsize=12)
    
    # Legend font size
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
