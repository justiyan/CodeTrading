# Transformer Trading Bot with Oanda API

A sophisticated trading bot that uses a Transformer neural network to predict forex price movements and execute trades through the Oanda API.

## Features

- **Transformer Model**: Advanced neural network architecture for time series prediction
- **Oanda API Integration**: Real-time market data and trade execution
- **Risk Management**: Position sizing, stop-loss, and take-profit management
- **Technical Indicators**: RSI, Bollinger Bands, Moving Averages
- **Configurable Parameters**: JSON-based configuration for easy customization
- **Logging**: Comprehensive logging for monitoring and debugging

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Oanda API

Create a `.env` file in the project root:

```env
OANDA_ACCOUNT_ID=your_account_id
OANDA_API_TOKEN=your_api_token
OANDA_ENVIRONMENT=practice  # or 'live' for real trading
```

### 3. Train the Model

Use the `Transformer_Trading.ipynb` notebook to train your model:

1. Load your forex data (EUR/USD recommended)
2. Run all cells to train the model
3. Save the model using the provided code at the end

### 4. Save the Trained Model

After training in the notebook, save the model for the bot:

```python
# In the notebook, after training:
from train_save_model import save_trained_model_from_notebook

save_trained_model_from_notebook(
    trained_model,
    "model.pth",
    scaler,
    "scaler.pkl"
)
```

## Usage

### Running the Trading Bot

```bash
python trading_bot.py --config bot_config.json --model model.pth --scaler scaler.pkl
```

### Configuration

Edit `bot_config.json` to customize trading parameters:

```json
{
    "instrument": "EUR_USD",
    "granularity": "H1",
    "lookback_candles": 100,
    "risk_per_trade": 0.01,
    "stop_loss_pips": 20,
    "risk_reward_ratio": 2.0,
    "max_position_size": 10000,
    "daily_trade_limit": 10,
    "confidence_threshold": 0.0005,
    "check_interval": 300,
    "device": "cpu"
}
```

## Project Structure

- `Transformer_Trading.ipynb` - Model training notebook
- `trading_bot.py` - Main bot orchestration
- `model_inference.py` - Model loading and signal generation
- `oanda_data.py` - Market data fetching
- `trade_manager.py` - Trade execution and management
- `oanda_config.py` - API configuration
- `bot_config.json` - Bot parameters

## Risk Warning

⚠️ **IMPORTANT**: This bot is for educational purposes. Trading forex involves substantial risk of loss. Always test thoroughly with a practice account before using real money.

## License

MIT