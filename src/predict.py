"""
Make predictions using trained phenotype classifier.
"""
import argparse
import pandas as pd
from utils import load_model, clean_data


def predict(model, data):
    """Make predictions on new data."""
    predictions = model.predict(data)
    probabilities = model.predict_proba(data)
    return predictions, probabilities


def main(model_path, data_path, output_path=None):
    """Main prediction pipeline."""
    print(f"Loading model from {model_path}...")
    model = load_model(model_path)
    
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    df_clean = clean_data(df)
    
    print("Making predictions...")
    predictions, probabilities = predict(model, df_clean)
    
    # Add predictions to dataframe
    df['predicted_label'] = predictions
    df['prediction_confidence'] = probabilities.max(axis=1)
    
    print(f"\nPredictions summary:")
    print(df['predicted_label'].value_counts())
    
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"\nResults saved to {output_path}")
    else:
        print("\nFirst 10 predictions:")
        print(df[['predicted_label', 'prediction_confidence']].head(10))
    
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict phenotypes using trained model')
    parser.add_argument('--model', type=str, default='models/model.pkl',
                        help='Path to trained model')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to data CSV for prediction')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save predictions CSV')
    
    args = parser.parse_args()
    main(args.model, args.data, args.output)
