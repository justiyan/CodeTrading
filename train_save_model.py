import torch
import pickle
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from model_inference import TimeSeriesTransformer, TradingSignalGenerator


def save_trained_model_from_notebook(
    notebook_model,
    filepath="model.pth",
    scaler=None,
    scaler_path="scaler.pkl"
):
    """
    Save a trained model from the notebook

    Args:
        notebook_model: The trained PyTorch model from notebook
        filepath: Path to save the model
        scaler: MinMaxScaler used for data preprocessing
        scaler_path: Path to save the scaler
    """
    # Model configuration
    config = {
        'feature_size': 9,
        'num_layers': 2,
        'd_model': 64,
        'nhead': 8,
        'dim_feedforward': 256,
        'dropout': 0.1,
        'seq_length': 30,
        'prediction_length': 1
    }

    # Save model with config
    torch.save({
        'model_state_dict': notebook_model.state_dict(),
        'model_config': config
    }, filepath)

    print(f"Model saved to {filepath}")

    # Save scaler if provided
    if scaler is not None:
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"Scaler saved to {scaler_path}")


# Example usage after training in notebook:
# save_trained_model_from_notebook(trained_model, "model.pth", scaler, "scaler.pkl")