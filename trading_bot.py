import time
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import json
import os

from oanda_config import OandaConfig
from oanda_data import OandaDataFetcher
from trade_manager import TradeManager
from model_inference import TradingSignalGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class TradingBot:
    """Main trading bot that orchestrates all components"""

    def __init__(
        self,
        config_file: str = "bot_config.json",
        model_path: str = "model.pth",
        scaler_path: Optional[str] = None
    ):
        # Load bot configuration
        self.load_config(config_file)

        # Initialize Oanda components
        self.oanda_config = OandaConfig.from_env()
        self.data_fetcher = OandaDataFetcher(self.oanda_config)
        self.trade_manager = TradeManager(self.oanda_config)

        # Initialize model
        self.signal_generator = TradingSignalGenerator(
            model_path=model_path,
            scaler_path=scaler_path,
            device=self.config.get("device", "cpu"),
            confidence_threshold=self.config.get("confidence_threshold", 0.0005)
        )

        # Trading state
        self.current_position = None
        self.last_signal_time = None
        self.trade_count = 0
        self.daily_trade_limit = self.config.get("daily_trade_limit", 10)

    def load_config(self, config_file: str):
        """Load bot configuration from JSON file"""
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                self.config = json.load(f)
        else:
            # Default configuration
            self.config = {
                "instrument": "EUR_USD",
                "granularity": "H1",
                "lookback_candles": 100,
                "risk_per_trade": 0.01,
                "stop_loss_pips": 20,
                "risk_reward_ratio": 2.0,
                "max_position_size": 10000,
                "daily_trade_limit": 10,
                "confidence_threshold": 0.0005,
                "check_interval": 300,  # 5 minutes
                "device": "cpu"
            }
            # Save default config
            with open(config_file, 'w') as f:
                json.dump(self.config, f, indent=4)
            logger.info(f"Created default configuration file: {config_file}")

    def check_trading_conditions(self) -> bool:
        """Check if trading conditions are met"""
        # Check daily trade limit
        if self.trade_count >= self.daily_trade_limit:
            logger.warning(f"Daily trade limit reached: {self.trade_count}/{self.daily_trade_limit}")
            return False

        # Check if minimum time has passed since last signal
        if self.last_signal_time:
            time_since_signal = datetime.now() - self.last_signal_time
            min_time = timedelta(hours=1)  # Minimum 1 hour between trades
            if time_since_signal < min_time:
                logger.debug(f"Too soon since last signal: {time_since_signal}")
                return False

        return True

    def fetch_and_prepare_data(self) -> tuple:
        """Fetch market data and prepare for model"""
        try:
            # Fetch candles
            df = self.data_fetcher.get_candles(
                instrument=self.config["instrument"],
                granularity=self.config["granularity"],
                count=self.config["lookback_candles"]
            )

            if df.empty:
                logger.error("No data fetched")
                return None, None

            # Add technical indicators
            df = self.data_fetcher.add_technical_indicators(df)

            # Get current price
            price_data = self.data_fetcher.get_latest_price(self.config["instrument"])
            current_price = (price_data["bid"] + price_data["ask"]) / 2

            return df, current_price

        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            return None, None

    def execute_trade(self, signal: Dict[str, Any]):
        """Execute trade based on signal"""
        try:
            # Get account details
            account = self.trade_manager.get_account_details()
            balance = float(account["balance"])

            # Get current price
            price_data = self.data_fetcher.get_latest_price(self.config["instrument"])

            if signal["signal"] == "BUY":
                entry_price = price_data["ask"]
            else:  # SELL
                entry_price = price_data["bid"]

            # Calculate stop loss and take profit
            stop_loss, take_profit = self.trade_manager.calculate_stop_loss_take_profit(
                entry_price=entry_price,
                direction=signal["signal"],
                stop_loss_pips=self.config["stop_loss_pips"],
                risk_reward_ratio=self.config["risk_reward_ratio"]
            )

            # Calculate position size
            position_size = self.trade_manager.calculate_position_size(
                account_balance=balance,
                risk_percentage=self.config["risk_per_trade"],
                stop_loss_pips=self.config["stop_loss_pips"]
            )

            # Apply max position size limit
            position_size = min(position_size, self.config["max_position_size"])

            # Adjust units based on direction
            if signal["signal"] == "SELL":
                position_size = -position_size

            # Place order
            logger.info(f"Placing {signal['signal']} order: {position_size} units at {entry_price}")
            logger.info(f"Stop Loss: {stop_loss}, Take Profit: {take_profit}")

            result = self.trade_manager.place_market_order(
                instrument=self.config["instrument"],
                units=position_size,
                stop_loss=stop_loss,
                take_profit=take_profit
            )

            # Update state
            self.current_position = signal["signal"]
            self.last_signal_time = datetime.now()
            self.trade_count += 1

            logger.info(f"Order placed successfully: {result}")
            return result

        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            return None

    def check_and_manage_positions(self):
        """Check and manage existing positions"""
        try:
            positions = self.trade_manager.get_open_positions()
            trades = self.trade_manager.get_open_trades()

            for trade in trades:
                logger.info(f"Open trade: {trade.instrument} - Units: {trade.units} - P&L: {trade.unrealized_pnl}")

                # Optional: Implement trailing stop or other position management
                # This is a placeholder for more sophisticated position management
                if trade.unrealized_pnl > 50:  # If profit > $50
                    # Move stop loss to break even
                    logger.info("Moving stop loss to break even")
                    # self.trade_manager.modify_trade(trade.id, stop_loss=trade.price)

        except Exception as e:
            logger.error(f"Error managing positions: {e}")

    def run_cycle(self):
        """Run one trading cycle"""
        logger.info("Starting trading cycle")

        # Check existing positions
        self.check_and_manage_positions()

        # Check trading conditions
        if not self.check_trading_conditions():
            return

        # Fetch and prepare data
        df, current_price = self.fetch_and_prepare_data()
        if df is None:
            return

        # Generate signal
        signal = self.signal_generator.generate_signal(df, current_price)
        logger.info(f"Generated signal: {signal}")

        # Execute trade if signal is not HOLD
        if signal["signal"] in ["BUY", "SELL"]:
            # Check if we already have a position
            positions = self.trade_manager.get_open_positions()
            instrument_positions = [p for p in positions if p.instrument == self.config["instrument"]]

            if not instrument_positions:
                # No existing position, execute trade
                self.execute_trade(signal)
            else:
                logger.info(f"Already have position in {self.config['instrument']}, skipping signal")

    def run(self):
        """Main bot loop"""
        logger.info("Starting Trading Bot")
        logger.info(f"Configuration: {self.config}")

        while True:
            try:
                # Reset daily trade count at midnight
                if datetime.now().hour == 0 and datetime.now().minute == 0:
                    self.trade_count = 0
                    logger.info("Daily trade count reset")

                # Run trading cycle
                self.run_cycle()

                # Wait before next cycle
                logger.info(f"Waiting {self.config['check_interval']} seconds before next cycle")
                time.sleep(self.config["check_interval"])

            except KeyboardInterrupt:
                logger.info("Bot stopped by user")
                break
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                time.sleep(60)  # Wait 1 minute before retrying

    def backtest(self, start_date: str, end_date: str):
        """Run backtest on historical data"""
        # This is a placeholder for backtesting functionality
        # Would need to implement historical data fetching and simulated trading
        pass


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Transformer Trading Bot")
    parser.add_argument("--config", default="bot_config.json", help="Bot configuration file")
    parser.add_argument("--model", default="model.pth", help="Model file path")
    parser.add_argument("--scaler", default=None, help="Scaler file path")
    parser.add_argument("--mode", default="live", choices=["live", "paper", "backtest"], help="Trading mode")

    args = parser.parse_args()

    # Create and run bot
    bot = TradingBot(
        config_file=args.config,
        model_path=args.model,
        scaler_path=args.scaler
    )

    if args.mode == "backtest":
        # Run backtest
        print("Backtest mode not yet implemented")
    else:
        # Run live/paper trading
        bot.run()


if __name__ == "__main__":
    main()