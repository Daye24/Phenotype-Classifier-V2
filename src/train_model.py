"""
Train Random Forest classifier for phenotype classification.
"""
import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from utils import load_data, clean_data, encode_labels, split_data, save_model


def train_random_forest(X_train, y_train, n_estimators=200, random_state=42):
    """Train Random Forest classifier."""
    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    rf.fit(X_train, y_train)
    return rf


def evaluate_model(model, X_test, y_test):
    """Evaluate model performance."""
    preds = model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)
    
    print(f'Accuracy: {accuracy:.4f}')
    print('\nClassification Report:')
    print(classification_report(y_test, preds))
    
    return preds


def plot_confusion_matrix(y_test, preds, save_path='models/confusion_matrix.png'):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Confusion matrix saved to {save_path}")
    plt.close()


def main(data_path, output_path='models/model.pkl'):
    """Main training pipeline."""
    print("Loading and preparing data...")
    df = load_data(data_path)
    df = clean_data(df)
    df = encode_labels(df, 'label')
    
    print("Splitting data...")
    X_train, X_test, y_train, y_test = split_data(df)
    print(f'Training set: {X_train.shape}, Test set: {X_test.shape}')
    
    print("\nTraining Random Forest...")
    model = train_random_forest(X_train, y_train)
    
    print("\nEvaluating model...")
    preds = evaluate_model(model, X_test, y_test)
    
    print("\nGenerating confusion matrix...")
    plot_confusion_matrix(y_test, preds)
    
    print("\nSaving model...")
    save_model(model, output_path)
    
    print("\n✅ Training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train phenotype classifier')
    parser.add_argument('--data', type=str, default='data/dataset.csv',
                        help='Path to training data CSV')
    parser.add_argument('--output', type=str, default='models/model.pkl',
                        help='Path to save trained model')
    
    args = parser.parse_args()
    main(args.data, args.output)
