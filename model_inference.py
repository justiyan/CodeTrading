import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, Optional, List
import pickle
import json

class TimeSeriesTransformer(nn.Module):
    """Same model architecture as in the notebook"""

    def __init__(
        self,
        feature_size=9,
        num_layers=2,
        d_model=64,
        nhead=8,
        dim_feedforward=256,
        dropout=0.1,
        seq_length=30,
        prediction_length=1
    ):
        super(TimeSeriesTransformer, self).__init__()

        self.input_fc = nn.Linear(feature_size, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_length, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu"
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, prediction_length)

    def forward(self, src):
        batch_size, seq_len, _ = src.shape
        src = self.input_fc(src)
        src = src + self.pos_embedding[:, :seq_len, :]
        src = src.permute(1, 0, 2)
        encoded = self.transformer_encoder(src)
        last_step = encoded[-1, :, :]
        out = self.fc_out(last_step)
        return out


class TradingSignalGenerator:
    """Generate trading signals from model predictions"""

    def __init__(
        self,
        model_path: str,
        scaler_path: Optional[str] = None,
        device: str = "cpu",
        confidence_threshold: float = 0.0005,
        feature_cols: Optional[List[str]] = None
    ):
        self.device = torch.device(device)
        self.confidence_threshold = confidence_threshold

        # Feature columns matching notebook
        if feature_cols is None:
            self.feature_cols = [
                'open', 'high', 'low', 'close',
                'rsi', 'bb_high', 'bb_low', 'ma_20', 'ma_20_slope'
            ]
        else:
            self.feature_cols = feature_cols

        # Load model
        self.model = self.load_model(model_path)
        self.model.eval()

        # Load or create scaler
        if scaler_path:
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
        else:
            self.scaler = MinMaxScaler()

    def load_model(self, model_path: str) -> TimeSeriesTransformer:
        """Load trained model"""
        checkpoint = torch.load(model_path, map_location=self.device)

        # Extract model configuration if saved
        if 'model_config' in checkpoint:
            config = checkpoint['model_config']
        else:
            # Default configuration
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

        model = TimeSeriesTransformer(**config)

        # Load state dict
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        model.to(self.device)
        return model

    def save_model(self, model: nn.Module, filepath: str, config: dict):
        """Save model with configuration"""
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_config': config
        }, filepath)

    def prepare_data(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare and scale data for model input"""
        # Select features
        data = df[self.feature_cols].values

        # Fit or transform with scaler
        if not hasattr(self.scaler, 'data_min_'):
            # Scaler not fitted yet
            data_scaled = self.scaler.fit_transform(data)
        else:
            data_scaled = self.scaler.transform(data)

        return data_scaled

    def predict_next_price(self, input_data: np.ndarray) -> Tuple[float, float]:
        """
        Predict next price from input sequence

        Args:
            input_data: Scaled input data (seq_length, num_features)

        Returns:
            (predicted_price, confidence)
        """
        with torch.no_grad():
            # Convert to tensor and add batch dimension
            x = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0)
            x = x.to(self.device)

            # Get prediction
            prediction = self.model(x).cpu().numpy()[0, 0]

            # Inverse transform to get actual price
            close_idx = self.feature_cols.index('close')
            dummy = np.zeros((1, len(self.feature_cols)))
            dummy[0, close_idx] = prediction
            predicted_price = self.scaler.inverse_transform(dummy)[0, close_idx]

            # Get current price (last close)
            current_dummy = np.zeros((1, len(self.feature_cols)))
            current_dummy[0, close_idx] = input_data[-1, close_idx]
            current_price = self.scaler.inverse_transform(current_dummy)[0, close_idx]

            # Calculate confidence based on price change
            price_change = (predicted_price - current_price) / current_price
            confidence = abs(price_change)

            return predicted_price, confidence

    def generate_signal(
        self,
        df: pd.DataFrame,
        current_price: float
    ) -> dict:
        """
        Generate trading signal from dataframe

        Returns:
            Dictionary with signal information
        """
        # Prepare data
        data_scaled = self.prepare_data(df)

        # Get last sequence
        if len(data_scaled) < 30:
            return {
                'signal': 'HOLD',
                'confidence': 0.0,
                'predicted_price': current_price,
                'reason': 'Insufficient data'
            }

        input_seq = data_scaled[-30:]

        # Get prediction
        predicted_price, confidence = self.predict_next_price(input_seq)

        # Calculate price change percentage
        price_change_pct = (predicted_price - current_price) / current_price

        # Generate signal
        signal = 'HOLD'
        reason = ''

        if confidence > self.confidence_threshold:
            if price_change_pct > 0.001:  # 0.1% threshold for BUY
                signal = 'BUY'
                reason = f'Predicted price increase: {price_change_pct:.4%}'
            elif price_change_pct < -0.001:  # 0.1% threshold for SELL
                signal = 'SELL'
                reason = f'Predicted price decrease: {price_change_pct:.4%}'
            else:
                reason = f'Price change too small: {price_change_pct:.4%}'
        else:
            reason = f'Low confidence: {confidence:.6f}'

        # Additional filters based on technical indicators
        last_row = df.iloc[-1]

        # RSI filter
        if 'rsi' in df.columns:
            if last_row['rsi'] > 70 and signal == 'BUY':
                signal = 'HOLD'
                reason += ' (RSI overbought)'
            elif last_row['rsi'] < 30 and signal == 'SELL':
                signal = 'HOLD'
                reason += ' (RSI oversold)'

        return {
            'signal': signal,
            'confidence': confidence,
            'predicted_price': predicted_price,
            'current_price': current_price,
            'price_change_pct': price_change_pct,
            'reason': reason,
            'rsi': last_row.get('rsi', None),
            'timestamp': df.index[-1] if not df.empty else None
        }