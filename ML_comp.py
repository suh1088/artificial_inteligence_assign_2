import time
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report

# ==========================================
# 1. Model Import
# ==========================================
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB

def main():
    # ==========================================
    # 2. Data Loading & Preprocessing (Common Pipeline)
    # ==========================================
    print("1. Loading MNIST dataset... (This may take a moment)")
    # Fetch MNIST data (784 dimensions)
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    
    X = mnist.data
    y = mnist.target.astype(int)

    # [Optimization] 
    # Using a subset of data for faster demonstration execution.
    # If you want to use the full dataset, comment out the following 3 lines.
    subset_size = 30000
    print(f"   - Subsampling dataset to {subset_size} samples for efficiency...")
    X = X[:subset_size]
    y = y[:subset_size]

    # Split Data (80% Train, 20% Test)
    print("2. Splitting data into Train/Test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scaling (Normalization to 0-1 range is crucial for SVM and MLP)
    print("3. Scaling features (MinMaxScaler)...")
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"   - Training Data Shape: {X_train_scaled.shape}")
    print(f"   - Test Data Shape:     {X_test_scaled.shape}")
    print("-" * 60)

    # ==========================================
    # 3. Model Definitions
    # ==========================================
    # List of (Name, Model Object) tuples
    models = [
        # Linear Regression for Classification (Least Squares)
        ("Linear Reg (Ridge)", RidgeClassifier()), 
        
        # Logistic Regression (Softmax)
        ("Logistic Regression", LogisticRegression(max_iter=1000, solver='lbfgs')), 
        
        # Support Vector Machine (RBF Kernel)
        ("SVM (RBF)", SVC(kernel='rbf', gamma='scale')), 
        
        # Multilayer Perceptron (Neural Network)
        ("MLP (Neural Net)", MLPClassifier(hidden_layer_sizes=(100,50), max_iter=300, random_state=42)), 
        
        # FDA (Fisher Discriminant Analysis) -> Implemented as LDA in sklearn
        ("FDA (LDA)", LinearDiscriminantAnalysis()), 
        
        # Graphical Model (Naive Bayes)
        ("Graphical (NaiveBayes)", GaussianNB()) 
    ]

    # ==========================================
    # 4. Training & Evaluation Loop
    # ==========================================
    results = []

    print(f"{'Model Name':<25} | {'Accuracy':<10} | {'Train Time (s)':<15} | {'Pred Time (s)':<15}")
    print("-" * 75)

    for name, model in models:
        # 1. Training Phase
        start_train = time.time()
        model.fit(X_train_scaled, y_train)
        end_train = time.time()
        train_time = end_train - start_train

        # 2. Prediction Phase
        start_pred = time.time()
        y_pred = model.predict(X_test_scaled)
        end_pred = time.time()
        pred_time = end_pred - start_pred

        # 3. Evaluation Phase
        acc = accuracy_score(y_test, y_pred)
        
        # Print row
        print(f"{name:<25} | {acc:.4f}     | {train_time:.4f}          | {pred_time:.4f}")

        # Store detailed results
        results.append({
            "Model": name,
            "Accuracy": acc,
            "Training Time": train_time,
            "Prediction Time": pred_time
        })

    # ==========================================
    # 5. Final Summary
    # ==========================================
    print("-" * 75)
    print("Comparison Completed.")
    
    # Optional: Convert to DataFrame for nicer display if needed
    # df_results = pd.DataFrame(results)
    # print(df_results)

if __name__ == "__main__":
    main()