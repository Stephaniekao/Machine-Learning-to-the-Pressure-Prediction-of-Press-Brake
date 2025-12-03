import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from scipy.stats import jarque_bera
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# 1. Read data
data = pd.read_excel(r"D:\論文\論文數據整理.xlsx")

numeric_features = ['厚度(mm)', '長度(mm)', '角度(°)']  # Numeric features
categorical_features = ['材料', '刀具', '下模']  # Categorical features
target = '壓力(Pa)'  # Target column

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
# Perform interpolation
data = data.interpolate(method='linear', axis=0)
# Check missing values after interpolation
print("Missing values after interpolation:")
print(data.isnull().sum())

# Outlier handling (using IQR range to detect outliers)
print(len(data))
print("Outlier handling")
for col in ['厚度(mm)', '角度(°)', '壓力(Pa)']:
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    data = data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]
    print(lower_bound, upper_bound)
    if ((data[col] < lower_bound) | (data[col] > upper_bound)).any():
        print("Outliers have been removed")
    else:
        print("No outliers")

# Check and remove duplicates
print("Removing duplicates")
print(len(data))
data = data.drop_duplicates()
print(len(data))
print(data)

print("One-hot encoding")
materials = ['SPCC', 'SUS', 'SUS_Tape']
tools = ['劍刀', '尖刀']
molds = ['10V', '8V', '6V']

# Create OneHotEncoder, explicitly specify category order
encoder = OneHotEncoder(categories=[materials, tools, molds], sparse_output=False)
encoded_features = encoder.fit_transform(data[categorical_features])
print(data[categorical_features])
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_features))
data = pd.concat([data.drop(columns=categorical_features).reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)

# Save the result to a file
data.to_csv(r"D:\論文\資料預處理\encoded_features_cleaned.txt", 
            sep='\t', index=False, encoding='utf-8')
# 4. Feature processing
print("Feature processing")
data_origin = data[['厚度(mm)', '長度(mm)', '角度(°)', '壓力(Pa)']]
# Normalization
scaler_minmax = MinMaxScaler()
data_minmax = scaler_minmax.fit_transform(data[['厚度(mm)', '長度(mm)', '角度(°)']])
data_minmax = pd.DataFrame(data_minmax, columns=['厚度(mm)', '長度(mm)', '角度(°)'])
print(data_minmax.describe())  # View data distribution
print(data_minmax.min())      # Should be 0
print(data_minmax.max())      # Should be 1

# Chinese column names mapped to English column names
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

    # Lower plot: MinMax Scaled data KDE
    sns.kdeplot(data_minmax[col], ax=axes[1], color='red', label='MinMax Scaled', linewidth=2)
    axes[1].set_title(f'{column_names[col]} - MinMax Scaled Distribution', fontsize=16, fontweight='bold')
    axes[1].set_xlabel(column_names[col], fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Probability Density', fontsize=14, fontweight='bold')
    axes[1].tick_params(axis='both', labelsize=12)
    axes[1].legend(fontsize=12)
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()
  
data_minmax.to_csv(r"D:\論文\資料預處理\data_standard_324.txt", 
                     sep='\t',   # Use tab as the delimiter
                     index=True,  # Do not save the index column
                     header=True,  # Keep the header row
                     encoding='utf-8')  # Ensure no garbled characters for Chinese

data_minmax_df = pd.DataFrame(data_minmax, columns=['厚度(mm)', '長度(mm)', '角度(°)'])
print(data)
encoded_df = data[['材料_SPCC', '材料_SUS','材料_SUS_Tape', '刀具_劍刀', '刀具_尖刀','下模_10V','下模_8V','下模_6V']]
print(encoded_df)
encoded_df.to_csv(r"D:\論文\資料預處理\data_standard_3241.txt", 
                     sep='\t',   # Use tab as the delimiter
                     index=True,  # Do not save the index column
                     header=True,  # Keep the header row
                     encoding='utf-8')  # Ensure no garbled characters for Chinese
merged_data = pd.concat([data_minmax_df, encoded_df, data['壓力(Pa)']], axis=1)  
# 3. Confirm merge result
print("Merged DataFrame info:")
print(merged_data.info())
print(merged_data.head())

# 4. Save as txt file
merged_data.to_csv(r"D:\論文\資料預處理\data_standard_3242.txt", 
                   sep='\t',   # Use tab as the delimiter
                   index=True,  # Do not save the index column
                   header=True,  # Keep the header row
                   encoding='utf-8')  # Ensure no garbled characters for Chinese

print("File successfully saved as data_standard_3242.txt")

joblib.dump(encoder, r"D:\python\encoder.pkl")
joblib.dump(scaler_minmax, r"D:\python\scaler.pkl")
print(encoder)
print(scaler_minmax)
print(encoder.get_feature_names_out(['材料', '刀具', '下模']))

# Split dataset
print("Data splitting")
print(merged_data)
X = merged_data.drop(columns='壓力(Pa)')
y = merged_data['壓力(Pa)']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Check the shape of the split data
print(f"Training set shape: X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"Testing set shape: X_test: {X_test.shape}, y_test: {y_test.shape}")

# Merge X_train and y_train
train_data = pd.concat([X_train, y_train], axis=1)

# Merge X_test and y_test
test_data = pd.concat([X_test, y_test], axis=1)

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
    
    # Original data distribution    
    sns.kdeplot(data_minmax[col], color='blue', label='Original Data', linewidth=2)
    # Training data distribution
    sns.kdeplot(X_train[col], color='green', label='Training Data', linewidth=2, linestyle='--')
    # Testing data distribution
    sns.kdeplot(X_test[col], color='red', label='Testing Data', linewidth=2, linestyle=':')
    
    # Increase and bolden title and labels
    plt.title(f'{column_names[col]} - Distribution Before and After Data Splitting',
              fontsize=16, fontweight='bold')
    plt.xlabel(column_names[col], fontsize=14, fontweight='bold')
    plt.ylabel('Probability Density', fontsize=14, fontweight='bold')
    
    # Axis tick label size
    plt.tick_params(axis='both', labelsize=12)
    
    # Legend font size
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
