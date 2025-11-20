#!/usr/bin/env python3
"""
Train a logistic regression classifier to distinguish between GLI and MET datasets
using auxiliary task features (area, shape, satellite, region).
"""

import json
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_prepare_data(aux_file):
    """
    Load auxiliary dataset and prepare features for classification.
    
    Returns:
        features: DataFrame with one-hot/k-hot encoded features
        labels: Array with GLI=0, MET=1 labels
        case_names: List of case names for reference
    """
    
    print(f"Loading auxiliary data from: {aux_file}")
    with open(aux_file, 'r') as f:
        aux_data = json.load(f)
    
    print(f"Loaded {len(aux_data)} samples")
    
    # Extract features and labels
    features_list = []
    labels = []
    case_names = []
    
    for entry in aux_data:
        seg_file = entry['seg_file']
        case_name = Path(seg_file).parent.name
        case_names.append(case_name)
        
        # Determine label (GLI=0, MET=1)
        if 'GLI' in seg_file:
            label = 0
        elif 'MET' in seg_file:
            label = 1
        else:
            raise ValueError(f"Unknown dataset type for {case_name}")
            
        labels.append(label)
        
        # Extract features for each label type
        label_features = {}
        
        for label_name, label_data in entry['labels'].items():
            if "Cavity" in label_name:
                continue  # Skip Resection Cavity label
            prefix = label_name.replace(' ', '_').replace('/', '_')
            
            # Area (one-hot encode)
            label_features[f'{prefix}_area'] = label_data['area']
            
            # Shape (one-hot encode) 
            label_features[f'{prefix}_shape'] = label_data['shape']
            
            # Satellite (one-hot encode)
            label_features[f'{prefix}_satellite'] = label_data['satellite']
            
            # Region (k-hot encode - list of regions)
            label_features[f'{prefix}_region'] = label_data['region']
        
        features_list.append(label_features)
    
    # Convert to DataFrame for easier processing
    features_df = pd.DataFrame(features_list)
    labels = np.array(labels)
    
    print(f"Feature columns: {list(features_df.columns)}")
    print(f"Labels: GLI={np.sum(labels==0)}, MET={np.sum(labels==1)}")
    
    return features_df, labels, case_names

def encode_features(features_df, feature_vocab=None):
    """
    One-hot encode area, shape, satellite and k-hot encode regions.
    If feature_vocab is provided, use it to ensure consistent encoding across splits.
    """
    
    encoded_features = []
    feature_names = []
    
    # Get all columns
    area_cols = [col for col in features_df.columns if '_area' in col]
    shape_cols = [col for col in features_df.columns if '_shape' in col]
    satellite_cols = [col for col in features_df.columns if '_satellite' in col]
    region_cols = [col for col in features_df.columns if '_region' in col]
    
    print(f"Found {len(area_cols)} area columns, {len(shape_cols)} shape columns, "
          f"{len(satellite_cols)} satellite columns, {len(region_cols)} region columns")
    
    # Build or use feature vocabulary
    if feature_vocab is None:
        feature_vocab = {}
        
        # Build vocabulary from training data
        for col in area_cols:
            values = features_df[col].values
            feature_vocab[col] = sorted(set(values))
            
        for col in shape_cols:
            values = features_df[col].values
            feature_vocab[col] = sorted(set(values))
            
        for col in satellite_cols:
            values = features_df[col].values
            feature_vocab[col] = sorted(set(values))
            
        for col in region_cols:
            regions_lists = features_df[col].values
            all_regions = set()
            for region_list in regions_lists:
                all_regions.update(region_list)
            feature_vocab[col] = sorted(list(all_regions))
    
    # One-hot encode area features
    for col in area_cols:
        values = features_df[col].values
        unique_vals = feature_vocab[col]
        print(f"{col} using values: {unique_vals}")
        
        for val in unique_vals:
            feature_name = f"{col}_{val}"
            feature_names.append(feature_name)
            encoded_features.append((values == val).astype(int))
    
    # One-hot encode shape features
    for col in shape_cols:
        values = features_df[col].values
        unique_vals = feature_vocab[col]
        print(f"{col} using values: {unique_vals}")
        
        for val in unique_vals:
            feature_name = f"{col}_{val}"
            feature_names.append(feature_name)
            encoded_features.append((values == val).astype(int))
    
    # One-hot encode satellite features
    for col in satellite_cols:
        values = features_df[col].values
        unique_vals = feature_vocab[col]
        print(f"{col} using values: {unique_vals}")
        
        for val in unique_vals:
            feature_name = f"{col}_{val}"
            feature_names.append(feature_name)
            encoded_features.append((values == val).astype(int))
    
    # K-hot encode region features
    for col in region_cols:
        regions_lists = features_df[col].values
        all_regions = feature_vocab[col]
        
        print(f"{col} using regions: {all_regions}")
        
        # Create binary features for each region
        for region in all_regions:
            feature_name = f"{col}_{region}"
            feature_names.append(feature_name)
            
            # Check if region is present in each sample's region list
            region_present = []
            for region_list in regions_lists:
                region_present.append(int(region in region_list))
            
            encoded_features.append(np.array(region_present))
    
    # Stack all features
    X = np.column_stack(encoded_features)
    
    print(f"Final feature matrix shape: {X.shape}")
    print(f"Feature names: {len(feature_names)}")
    
    return X, feature_names, feature_vocab

def train_and_evaluate_classifier(X_train, y_train, X_test, y_test, feature_names, random_state=42):
    """
    Train logistic regression and evaluate performance.
    """
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Training GLI/MET: {np.sum(y_train==0)}/{np.sum(y_train==1)}")
    print(f"Test GLI/MET: {np.sum(y_test==0)}/{np.sum(y_test==1)}")
    
    # Train logistic regression
    print("\nTraining logistic regression classifier...")
    clf = LogisticRegression(random_state=random_state, max_iter=1000)
    clf.fit(X_train, y_train)
    
    # Make predictions
    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)
    y_train_proba = clf.predict_proba(X_train)[:, 1]
    y_test_proba = clf.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    
    train_f1 = f1_score(y_train, y_train_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    
    train_auc = roc_auc_score(y_train, y_train_proba)
    test_auc = roc_auc_score(y_test, y_test_proba)
    
    print(f"\n=== RESULTS ===")
    print(f"Training Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Training F1: {train_f1:.4f}")
    print(f"Test F1: {test_f1:.4f}")
    print(f"Training AUC: {train_auc:.4f}")
    print(f"Test AUC: {test_auc:.4f}")
    
    # Classification report
    print(f"\n=== CLASSIFICATION REPORT (Test Set) ===")
    target_names = ['GLI', 'MET']
    print(classification_report(y_test, y_test_pred, target_names=target_names))
    
    # Confusion matrix
    print(f"\n=== CONFUSION MATRIX (Test Set) ===")
    cm = confusion_matrix(y_test, y_test_pred)
    print("     Predicted")
    print("     GLI  MET")
    print(f"GLI  {cm[0,0]:3d}  {cm[0,1]:3d}")
    print(f"MET  {cm[1,0]:3d}  {cm[1,1]:3d}")
    
    # Feature importance
    print(f"\n=== TOP 10 MOST IMPORTANT FEATURES ===")
    feature_importance = np.abs(clf.coef_[0])
    top_indices = np.argsort(feature_importance)[-10:][::-1]
    
    for i, idx in enumerate(top_indices):
        print(f"{i+1:2d}. {feature_names[idx]:40s} {feature_importance[idx]:8.4f}")
    
    return {
        'classifier': clf,
        'train_metrics': {'accuracy': train_acc, 'f1': train_f1, 'auc': train_auc},
        'test_metrics': {'accuracy': test_acc, 'f1': test_f1, 'auc': test_auc},
        'feature_importance': feature_importance,
        'feature_names': feature_names,
        'confusion_matrix': cm,
        'y_test': y_test,
        'y_test_pred': y_test_pred,
        'y_test_proba': y_test_proba
    }

def save_results(results, output_file):
    """Save results to JSON file."""
    
    # Convert numpy arrays to lists for JSON serialization
    results_json = {
        'train_metrics': results['train_metrics'],
        'test_metrics': results['test_metrics'],
        'feature_importance': results['feature_importance'].tolist(),
        'feature_names': results['feature_names'],
        'confusion_matrix': results['confusion_matrix'].tolist(),
        'y_test': results['y_test'].tolist(),
        'y_test_pred': results['y_test_pred'].tolist(),
        'y_test_proba': results['y_test_proba'].tolist()
    }
    
    with open(output_file, 'w') as f:
        json.dump(results_json, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Train GLI vs MET classifier using auxiliary features')
    parser.add_argument('--train_file',
                       default='brats_gli_met_3d_vqa_subjTrue_train_aux_combined_v11_seed0.json',
                       help='Path to training auxiliary dataset file')
    parser.add_argument('--val_file',
                       default='brats_gli_met_3d_vqa_subjTrue_val_aux_combined_v11_seed0.json',
                       help='Path to validation auxiliary dataset file')
    parser.add_argument('--test_file',
                       default='brats_gli_met_3d_vqa_subjTrue_test_aux_combined_v11_seed0.json',
                       help='Path to test auxiliary dataset file')
    parser.add_argument('--output_file',
                       default='gli_met_classification_results.json',
                       help='Output file for classification results')
    parser.add_argument('--random_state', type=int, default=42,
                       help='Random state for reproducibility')
    
    args = parser.parse_args()
    
    # Check if input files exist
    for file_path, file_name in [(args.train_file, 'Training'), (args.val_file, 'Validation'), (args.test_file, 'Test')]:
        if not Path(file_path).exists():
            print(f"Error: {file_name} file not found: {file_path}")
            return
    
    print("=== LOADING TRAINING DATA ===")
    train_features_df, train_labels, train_case_names = load_and_prepare_data(args.train_file)
    
    print("\n=== LOADING VALIDATION DATA ===")
    val_features_df, val_labels, val_case_names = load_and_prepare_data(args.val_file)
    
    print("\n=== LOADING TEST DATA ===")
    test_features_df, test_labels, test_case_names = load_and_prepare_data(args.test_file)
    
    # Encode training features and build vocabulary
    print("\n=== ENCODING TRAINING FEATURES ===")
    X_train, feature_names, feature_vocab = encode_features(train_features_df)
    
    # Encode validation features using training vocabulary
    print("\n=== ENCODING VALIDATION FEATURES ===")
    X_val, _, _ = encode_features(val_features_df, feature_vocab)
    
    # Encode test features using training vocabulary
    print("\n=== ENCODING TEST FEATURES ===")
    X_test, _, _ = encode_features(test_features_df, feature_vocab)
    
    # Train and evaluate classifier
    print("\n=== TRAINING AND EVALUATION ===")
    results = train_and_evaluate_classifier(
        X_train, train_labels, X_test, test_labels, feature_names, 
        random_state=args.random_state
    )
    
    # Optional: Also evaluate on validation set
    print(f"\n=== VALIDATION SET EVALUATION ===")
    val_pred = results['classifier'].predict(X_val)
    val_proba = results['classifier'].predict_proba(X_val)[:, 1]
    
    val_acc = accuracy_score(val_labels, val_pred)
    val_f1 = f1_score(val_labels, val_pred)
    val_auc = roc_auc_score(val_labels, val_proba)
    
    print(f"Validation Accuracy: {val_acc:.4f}")
    print(f"Validation F1: {val_f1:.4f}")
    print(f"Validation AUC: {val_auc:.4f}")
    
    # Add validation metrics to results
    results['val_metrics'] = {'accuracy': val_acc, 'f1': val_f1, 'auc': val_auc}
    
    # Save results
    save_results(results, args.output_file)

if __name__ == "__main__":
    main()