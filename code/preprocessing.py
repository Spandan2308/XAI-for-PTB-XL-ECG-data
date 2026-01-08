import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wfdb
import ast
from sklearn.preprocessing import MultiLabelBinarizer

def ecg_data_array(dir, df):
    
    data = np.array([wfdb.rdsamp(os.path.join(dir, f))[0] for f in df.filename_lr])
    np.save('preprocessed_ecg_data.npy', data) 
    print("Data saved successfully!")

    return data

TARGET = {"NORM","MI","STTC","CD","HYP"}

def create_labels(base_dir, df):
    
    # Load scp_statements.csv for diagnostic aggregation
    agg_df = pd.read_csv(os.path.join(base_dir, 'scp_statements.csv'), index_col=0)
    agg_df = agg_df[agg_df.diagnostic == 1]

    final_labels = []

    #Iterate through every row's dictionary
    for row_dict in df["scp_codes"]:
        tmp = set()
        for key in row_dict.keys():
            if key in agg_df.index:
                cls = agg_df.loc[key].diagnostic_class
                if cls in TARGET:
                    tmp.add(cls)
        final_labels.append(list(tmp))
    return final_labels

def plot_ecg(ecg_data, leads, index):
    """
    Visualizes the ECG signal for a single sample (patient) with all 12 leads.
    """
    ecg_sample = ecg_data[index]
    
    fig, axes = plt.subplots(12, 1, figsize=(14, 12), sharex=True)
    
    for i in range(12):
        axes[i].plot(ecg_sample[:, i], label=leads[i])
        axes[i].set_title(f"Lead: {leads[i]}")
        axes[i].legend(loc="upper right")
        
    plt.xlabel("Time (samples)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.show()

def split_data_by_stratified_fold(data, df):
    """
    Split data using PTB-XL stratified folds.
    """
    X_data = data  
    y_data = df["diagnostic_superclass"] 
    
    # Split the data into train, validation, and test based on strat_fold
    # first creates list of T/F ([T,F,F,...]) and use it as mask to keep rows from array corresponding to True.

    X_train = X_data[df.strat_fold < 9] 
    y_train = y_data[df.strat_fold < 9]

    X_val = X_data[df.strat_fold == 9]
    y_val = y_data[df.strat_fold == 9]

    X_test = X_data[df.strat_fold == 10]
    y_test = y_data[df.strat_fold == 10]

    return X_train, X_val, X_test, y_train, y_val, y_test
    
def normalize_data(X_train, X_val, X_test):
    
    #calculating mean and std of the training set (per lead)
    mean_train = np.mean(X_train, axis=(0, 1))  
    std_train = np.std(X_train, axis=(0, 1)) 
    
    X_train_normalized = (X_train - mean_train) / (std_train + 1e-6)
    
    #normalizing validation and test data using the training statistics
    X_val_normalized = (X_val - mean_train) / (std_train + 1e-6)
    X_test_normalized = (X_test - mean_train) / (std_train + 1e-6)
    
    return X_train_normalized, X_val_normalized, X_test_normalized

base_dir = "ptb-xl-1.0.3"
db_csv = "ptbxl_database.csv"
leads = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

db_path = os.path.join(base_dir, db_csv)

df = pd.read_csv(db_path, index_col='ecg_id')
df["scp_codes"] = df["scp_codes"].apply(lambda x: ast.literal_eval(x)) #convert string in csv into python dictionary
#print(df["scp_codes"])

#data = ecg_data_array(base_dir, df)

X = np.load('preprocessed_ecg_data.npy')  # Loads the data back
print("Data loaded successfully!")
#print(X.shape)

#plot_ecg(X, leads, 0)

labels = create_labels(base_dir, df)

#one hot encoding the labels and adding it as new column in main df
mlb = MultiLabelBinarizer(classes=["NORM","MI","STTC","CD","HYP"])
labels_encoded = mlb.fit_transform(labels)
df["diagnostic_superclass"] = list(labels_encoded)
print(df["diagnostic_superclass"])

X_train, X_val, X_test, y_train, y_val, y_test = split_data_by_stratified_fold(X, df)

y_train = np.vstack(y_train.values).astype(np.float32)
y_val   = np.vstack(y_val.values).astype(np.float32)
y_test  = np.vstack(y_test.values).astype(np.float32)


print(f"Training data shape: {X_train.shape}, Labels shape: {y_train.shape}")
print(f"Validation data shape: {X_val.shape}, Labels shape: {y_val.shape}")
print(f"Test data shape: {X_test.shape}, Labels shape: {y_test.shape}")

X_train_normalized, X_val_normalized, X_test_normalized = normalize_data(X_train, X_val, X_test)

for name, y in [("train", y_train), ("val", y_val), ("test", y_test)]:
    z = (y.sum(axis=1)==0).sum()
    print(name, "all-zero:", z, "/", y.shape[0], f"({z/y.shape[0]:.2%})")

print("mlb classes:", mlb.classes_)
print("y_train sums min/max:", y_train.sum(1).min(), y_train.sum(1).max())
print("y_train label prevalence:", y_train.mean(0))


np.save('X_train_normalized.npy', X_train_normalized)
np.save('X_val_normalized.npy', X_val_normalized)
np.save('X_test_normalized.npy', X_test_normalized)
np.save('y_train.npy', y_train)
np.save('y_val.npy', y_val)
np.save('y_test.npy', y_test)