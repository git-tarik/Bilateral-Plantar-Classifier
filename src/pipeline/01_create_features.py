# Bilateral-Plantar-Classifier — All rights reserved.
# Copyright (c) 2025 <Your Name/Team>. See LICENSE.
# NOTE: Do not commit human-subject data. See DATA_POLICY.md.


import pandas as pd
import numpy as np
import glob
import os
from scipy.signal import find_peaks
from scipy.integrate import simpson

def find_file_pairs(path='gait_data'):
    """Finds matching pairs of left and right foot data files."""
    pairs = {}
    all_files = [f for f in glob.glob(os.path.join(path, "*.csv")) if "master_features" not in f]
    
    for file in all_files:
        basename = os.path.basename(file)
        parts = basename.replace('.csv', '').split('_')
        
        if len(parts) >= 2:
            subject_id = parts[0]
            foot = parts[1]
            if subject_id not in pairs: pairs[subject_id] = {}
            if foot == 'left': pairs[subject_id]['left_file'] = file
            elif foot == 'right': pairs[subject_id]['right_file'] = file
                
    complete_pairs = [pair for pair in pairs.values() if 'left_file' in pair and 'right_file' in pair]
    print(f"Found {len(complete_pairs)} complete left/right file pairs.")
    return complete_pairs

def extract_bilateral_features(df_left, df_right):
    """
    Extracts features using the 'No' column. This version is designed to be
    resilient and avoid skipping files by handling cases with few detected steps.
    """
    df_left = df_left.set_index('No')
    df_right = df_right.set_index('No')
    
    if 'Timestamp' in df_left.columns: df_left = df_left.drop('Timestamp', axis=1)
    if 'Timestamp' in df_right.columns: df_right = df_right.drop('Timestamp', axis=1)
    
    df_left = df_left.rename(columns=lambda c: f"{c}_L")
    df_right = df_right.rename(columns=lambda c: f"{c}_R")
    
    df = pd.concat([df_left, df_right], axis=1).dropna()
    if df.empty: return None

    heel_strikes_L, _ = find_peaks(df['HFR4_L'], height=0.1, distance=5)
    heel_strikes_R, _ = find_peaks(df['HFR4_R'], height=0.1, distance=5)

    features = {}
    
    duration_rows = df.index[-1] - df.index[0]
    approx_hz = 30 
    duration_sec = duration_rows / approx_hz if approx_hz > 0 else 0
    
    # --- CHANGE: Added Failsafe Logic for Step Calculation ---
    # Process left foot if steps are detected, otherwise default to 0
    if len(heel_strikes_L) > 1:
        features['cadence_L'] = (len(heel_strikes_L) / duration_sec) * 60 if duration_sec > 0 else 0
        step_times_L = np.diff(heel_strikes_L)
        features['step_time_mean_L'] = np.mean(step_times_L)
        features['step_time_std_L'] = np.std(step_times_L)
        features['step_time_cv_L'] = (features['step_time_std_L'] / features['step_time_mean_L']) if features['step_time_mean_L'] > 0 else 0
    else:
        features.update({'cadence_L': 0, 'step_time_mean_L': 0, 'step_time_std_L': 0, 'step_time_cv_L': 0})

    # Process right foot if steps are detected, otherwise default to 0
    if len(heel_strikes_R) > 1:
        features['cadence_R'] = (len(heel_strikes_R) / duration_sec) * 60 if duration_sec > 0 else 0
        step_times_R = np.diff(heel_strikes_R)
        features['step_time_mean_R'] = np.mean(step_times_R)
        features['step_time_std_R'] = np.std(step_times_R)
        features['step_time_cv_R'] = (features['step_time_std_R'] / features['step_time_mean_R']) if features['step_time_mean_R'] > 0 else 0
    else:
        features.update({'cadence_R': 0, 'step_time_mean_R': 0, 'step_time_std_R': 0, 'step_time_cv_R': 0})
        
    # Average cadence
    features['cadence'] = (features['cadence_L'] + features['cadence_R']) / 2

    # PTI and Asymmetry are calculated regardless
    pti_L = sum(simpson(df[col].fillna(0), x=df.index) for col in df.columns if col.endswith('_L'))
    pti_R = sum(simpson(df[col].fillna(0), x=df.index) for col in df.columns if col.endswith('_R'))
    features['gait_asymmetry_index'] = (pti_L - pti_R) / (pti_L + pti_R) if (pti_L + pti_R) > 0 else 0

    return features

def label_pattern(features):
    """Applies lenient rules to classify gait."""
    if not features: return "Uncertain", "Feature extraction failed"
    
    abnormality_score = 0
    reasons = []

    cadence_lower_bound = 60 # Lenient lower bound
    cadence_upper_bound = 150 # Lenient upper bound
    # Cadence of 0 means no steps were detected, which is definitely abnormal
    if not (cadence_lower_bound < features.get('cadence', 0) < cadence_upper_bound):
        abnormality_score += 1
        reasons.append(f"Cadence_out_of_range ({features.get('cadence', 0):.1f})")

    asymmetry_threshold = 0.40
    if abs(features.get('gait_asymmetry_index', 0)) > asymmetry_threshold:
        abnormality_score += 1
        reasons.append(f"High_Asymmetry ({features.get('gait_asymmetry_index', 0):.2f})")

    instability_threshold = 0.30
    if features.get('step_time_cv_L', 0) > instability_threshold or features.get('step_time_cv_R', 0) > instability_threshold:
        abnormality_score += 1
        reasons.append("High_Step_Variability")

    # A single minor deviation is OK. Two or more suggests abnormality.
    if abnormality_score >= 1: # Let's be a bit stricter now
        return "Abnormal", ", ".join(reasons)
    else:
        return "Normal", "Within lenient parameters"

def main():
    print("--- STEP 1 & 2: Loading Data and Creating Features ---")
    file_pairs = find_file_pairs()
    if not file_pairs:
        print("\nNo file pairs found.")
        return
        
    all_features = []
    for pair in file_pairs:
        try:
            print(f"Processing pair: {os.path.basename(pair['left_file'])} & {os.path.basename(pair['right_file'])}")
            
            df_l = pd.read_csv(pair['left_file'])
            df_r = pd.read_csv(pair['right_file'])

            features = extract_bilateral_features(df_l, df_r)
            
            if features:
                label, reason = label_pattern(features)
                features['label'] = label
                features['reason_for_label'] = reason
                features['source_left'] = os.path.basename(pair['left_file'])
                features['source_right'] = os.path.basename(pair['right_file'])
                all_features.append(features)
            else:
                # This should rarely happen now
                print(f"  -> Skipping pair, data was empty after merging.")
        except Exception as e:
            print(f"  -> ERROR processing pair: {e}")

    if not all_features:
        print("\nFeature extraction complete, but no valid features were generated.")
        return
        
    master_df = pd.DataFrame(all_features)
    master_df = master_df.dropna()
    master_dataset_path = 'master_features_dataset.csv'
    master_df.to_csv(master_dataset_path, index=False)
    
    print(f"\nSUCCESS! Master feature dataset created at: '{master_dataset_path}'")
    print(f"Total valid samples processed: {len(master_df)}")
    if 'label' in master_df:
        print("Label distribution:\n", master_df['label'].value_counts())

if __name__ == "__main__":

    main()
