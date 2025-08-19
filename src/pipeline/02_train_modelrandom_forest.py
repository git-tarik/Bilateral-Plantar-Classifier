# Save this file as: 02_train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os

def main():
    """Main function to train and evaluate the model."""
    master_dataset_path = 'master_features_dataset.csv'
    
    # --- STEP 3: Loading Features ---
    print("--- STEP 3: Loading Features ---")
    if not os.path.exists(master_dataset_path):
        print(f"ERROR: '{master_dataset_path}' not found.")
        print("Please run '01_create_features.py' first to generate the feature dataset.")
        return
        
    df = pd.read_csv(master_dataset_path)
    print(f"Successfully loaded {len(df)} samples from '{master_dataset_path}'.")
    
    # Check if there is enough data to train a model
    if len(df) < 10 or len(df['label'].unique()) < 2:
        print("\nERROR: Not enough data or only one class present in the dataset to train a meaningful model.")
        return
    
    # --- STEP 4: Model Training & Evaluation ---
    print("\n--- STEP 4: Model Training & Evaluation ---")

    # Prepare data for Scikit-learn
    # X contains all the numerical features; y contains the labels we want to predict.
    X = df.drop(['label', 'reason_for_label', 'source_left', 'source_right'], axis=1)
    y = df['label']

    # Split data: 70% for training, 30% for testing
    # stratify=y ensures that the proportion of labels is the same in train and test sets.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples.")

    # Scale numerical features - this helps the model perform better
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize and train the classifier
    # class_weight='balanced' tells the model to pay more attention to the minority class ('Abnormal')
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train_scaled, y_train)
    
    # --- Evaluation ---
    print("\n--- MODEL PERFORMANCE ---")
    y_pred = model.predict(X_test_scaled)
    
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2%}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Display a visual Confusion Matrix
    print("Displaying Confusion Matrix plot...")
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()

if __name__ == "__main__":
    main()