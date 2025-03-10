import pandas as pd
import numpy as np
from glob import glob
import os
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_file(file_path):
    """Load and preprocess a single CSV file."""
    print(f"Loading file: {file_path}")
    df = pd.read_csv(file_path)
    
    # Drop EID and timestamp columns as per requirements
    df = df.drop(['EID', 'time'], axis=1)
    
    # Check for NaN values
    if df.isna().any().any():
        print(f"Warning: Found NaN values in {file_path}")
        df = df.fillna(0)  # Fill NaN values with 0
    
    # Check for infinite values
    if np.isinf(df.values).any():
        print(f"Warning: Found infinite values in {file_path}")
        df = df.replace([np.inf, -np.inf], 0)  # Replace infinite values with 0
    
    # Calculate additional features
    # Magnitude of acceleration
    df['magnitude'] = np.sqrt(df['Xvalue']**2 + df['Yvalue']**2 + df['Zvalue']**2)
    
    # Simple statistics instead of rolling ones
    df['mean'] = df[['Xvalue', 'Yvalue', 'Zvalue']].mean(axis=1)
    df['std'] = df[['Xvalue', 'Yvalue', 'Zvalue']].std(axis=1)
    
    # Check the range of values
    print(f"Value ranges for {file_path}:")
    print(df.describe())
    
    return df.values

def prepare_dataset(data_dir, scaler=None):
    """Prepare dataset from a directory of CSV files."""
    all_data = []
    all_labels = []
    
    # Process all files in the directory
    print(f"Looking for CSV files in: {os.path.abspath(data_dir)}")
    file_pattern = os.path.join(data_dir, "*.csv")
    files = glob(file_pattern)
    print(f"Found {len(files)} CSV files")
    
    # Process all files in the directory
    for file_path in files:
        # Extract label from filename
        filename = os.path.basename(file_path)
        label = 1 if filename.startswith("UserA") else 0  # UserA = 1, UserB = 0
        print(f"Processing {filename} with label {label}")
        
        try:
            # Load and preprocess file
            data = load_and_preprocess_file(file_path)
            
            # Check for NaN or infinite values in processed data
            if np.isnan(data).any() or np.isinf(data).any():
                print(f"Warning: Found NaN or infinite values in processed data for {filename}")
                continue
            
            # Store data and label
            all_data.append(data)
            all_labels.extend([label] * len(data))
            print(f"Successfully processed {filename}")
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
    
    if not all_data:
        raise ValueError(f"No data was loaded from {data_dir}. Check if the directory contains valid CSV files.")
    
    # Concatenate all data
    X = np.vstack(all_data)
    y = np.array(all_labels)
    
    print(f"Final dataset shape: X={X.shape}, y={y.shape}")
    
    # Additional checks on the final dataset
    print("\nFinal dataset statistics:")
    print("X mean:", np.mean(X))
    print("X std:", np.std(X))
    print("X min:", np.min(X))
    print("X max:", np.max(X))
    
    # Fit or transform with scaler
    if scaler is None:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    else:
        X = scaler.transform(X)
    
    # Check the scaled data
    print("\nScaled dataset statistics:")
    print("X mean:", np.mean(X))
    print("X std:", np.std(X))
    print("X min:", np.min(X))
    print("X max:", np.max(X))
    
    return X, y, scaler 