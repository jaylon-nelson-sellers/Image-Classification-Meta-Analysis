import time
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score,balanced_accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
def convert_data_to_tensors(X_train, X_test, y_train,image_bool=False):
    """ Convert numpy arrays to PyTorch tensors.

    Args:
        X_train (np.array): Training features.
        X_test (np.array): Testing features.
        y_train (np.array): Training labels.

    Returns:
        tuple: Tuple containing tensors for training features, testing features, and training labels.
    """
    # Check device availability (GPU or CPU)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Check if the data is a pandas DataFrame and convert to NumPy if true

    if isinstance(X_train, pd.DataFrame):
        X_train = X_train.to_numpy()
    if isinstance(X_test, pd.DataFrame):
        X_test = X_test.to_numpy()
    if isinstance(y_train, pd.DataFrame):
        y_train = y_train.to_numpy()

    if isinstance(X_train, pd.Series):
        X_train = X_train.to_numpy()
    if isinstance(X_test, pd.Series):
        X_test = X_test.to_numpy()
    if isinstance(y_train, pd.Series):
        y_train = y_train.to_numpy()
    # Ensure data is numpy array before converting to tensor
    if isinstance(X_train, np.ndarray) and isinstance(X_test, np.ndarray) and isinstance(y_train, np.ndarray):
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32, device=device)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32, device=device)
        if image_bool:
            y_train_tensor = torch.tensor(y_train, dtype=torch.float32, device=device)
        else:
            y_train_tensor = torch.tensor(y_train, dtype=torch.float32, device=device)
        return X_train_tensor, X_test_tensor, y_train_tensor
    else:
        raise TypeError("Input data should be either NumPy arrays or pandas DataFrames.")

def evaluate_image_test(model,dataset_data,dl,tensor_check=False,round_digits=4):
    X_train, X_test, y_train, y_test = dataset_data
    if tensor_check:
        X_train, X_test, y_traiN =convert_data_to_tensors(X_train, X_test, y_train,image_bool=not True)
    start = time.time()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    end = time.time()

    time_taken = end - start
    acc = accuracy_score(y_test, predictions)
    balanced_acc = balanced_accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average="micro")
    recall = recall_score(y_test, predictions, average="micro")
    f1 = f1_score(y_test, predictions, average="micro")
    cm = confusion_matrix(y_test, predictions)
    print(acc)

    results = [
        time_taken,
        round(acc, round_digits),
        round(balanced_acc, round_digits),
        round(precision, round_digits),
        round(recall, round_digits),
        round(f1, round_digits),
        cm
    ]
    dl.save_info(model, results)