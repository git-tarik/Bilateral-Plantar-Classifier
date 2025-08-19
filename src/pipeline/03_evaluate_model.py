# Save this file as: 03_evaluate_model.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc

def main():
    """
    Main function to load features, train a model, and generate
    advanced performance visualizations.
    """
    master_dataset_path = 'master_features_dataset.csv'
    
    # Create a directory to save the plots
    output_plot_dir = 'model_performance_plots'
    os.makedirs(output_plot_dir, exist_ok=True)

    # --- Step 1: Loading Features ---
    print("--- STEP 1: Loading Features ---")
    if not os.path.exists(master_dataset_path):
        print(f"ERROR: '{master_dataset_path}' not found.")
        return
        
    df = pd.read_csv(master_dataset_path)
    print(f"Successfully loaded {len(df)} samples.")
    
    # --- Step 2: Data Preparation & Model Training ---
    print("\n--- STEP 2: Training Model ---")
    X = df.drop(['label', 'reason_for_label', 'source_left', 'source_right'], axis=1)
    y = df['label']

    # Convert string labels to numbers (0 for Abnormal, 1 for Normal)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1] # Probabilities for the 'Normal' class

    print("\n--- STEP 3: Generating Performance Visualizations ---")

    # --- VISUALIZATION 1: Enhanced Confusion Matrix ---
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
    plt.title('Confusion Matrix', fontsize=16)
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(output_plot_dir, '01_confusion_matrix.png'))
    ### CHANGE ### Replaced emoji with plain text
    print("1. Confusion Matrix plot saved.")
    plt.close()

    # --- VISUALIZATION 2: Classification Report Heatmap ---
    report = classification_report(y_test, y_pred, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    plt.figure(figsize=(10, 6))
    sns.heatmap(df_report.iloc[:-1, :].T, annot=True, cmap='viridis')
    plt.title('Classification Report', fontsize=16)
    plt.savefig(os.path.join(output_plot_dir, '02_classification_report.png'))
    ### CHANGE ### Replaced emoji with plain text
    print("2. Classification Report heatmap saved.")
    plt.close()

    # --- VISUALIZATION 3: Feature Importance Chart ---
    importances = model.feature_importances_
    feature_names = X.columns
    feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
    feature_importance_df = feature_importance_df.sort_values('importance', ascending=False)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=feature_importance_df)
    plt.title('Feature Importance', fontsize=16)
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig(os.path.join(output_plot_dir, '03_feature_importance.png'))
    ### CHANGE ### Replaced emoji with plain text
    print("3. Feature Importance plot saved.")
    plt.close()

    # --- VISUALIZATION 4: ROC Curve and AUC ---
    # Convert y_test to binary format for ROC curve
    y_test_binary = (y_test == 'Normal').astype(int)
    fpr, tpr, _ = roc_curve(y_test_binary, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=16)
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_plot_dir, '04_roc_curve.png'))
    ### CHANGE ### Replaced emoji with plain text
    print("4. ROC Curve plot saved.")
    plt.close()
    
    print(f"\nSUCCESS! All plots have been saved to the '{output_plot_dir}' folder.")

if __name__ == "__main__":
    main()