import requests
import json
from typing import Optional, Dict, List, Any
from datetime import datetime
from dataclasses import dataclass
from oanda_config import OandaConfig

@dataclass
class Trade:
    """Trade data structure"""
    id: str
    instrument: str
    units: float
    price: float
    unrealized_pnl: float
    realized_pnl: float
    state: str
    open_time: datetime

@dataclass
class Position:
    """Position data structure"""
    instrument: str
    units: float
    unrealized_pnl: float
    average_price: float

class TradeManager:
    """Manages trades and positions with Oanda API"""

    def __init__(self, config: OandaConfig):
        self.config = config

    def get_account_details(self) -> Dict[str, Any]:
        """Get account details including balance and margin"""
        endpoint = f"{self.config.base_url}/v3/accounts/{self.config.account_id}"

        response = requests.get(endpoint, headers=self.config.headers)

        if response.status_code != 200:
            raise Exception(f"Failed to get account details: {response.text}")

        return response.json()["account"]

    def get_open_positions(self) -> List[Position]:
        """Get all open positions"""
        endpoint = f"{self.config.base_url}/v3/accounts/{self.config.account_id}/openPositions"

        response = requests.get(endpoint, headers=self.config.headers)

        if response.status_code != 200:
            raise Exception(f"Failed to get positions: {response.text}")

        positions = []
        for pos in response.json().get("positions", []):
            if pos["long"]["units"] != "0":
                positions.append(Position(
                    instrument=pos["instrument"],
                    units=float(pos["long"]["units"]),
                    unrealized_pnl=float(pos["long"]["unrealizedPL"]),
                    average_price=float(pos["long"]["averagePrice"])
                ))
            if pos["short"]["units"] != "0":
                positions.append(Position(
                    instrument=pos["instrument"],
                    units=float(pos["short"]["units"]),
                    unrealized_pnl=float(pos["short"]["unrealizedPL"]),
                    average_price=float(pos["short"]["averagePrice"])
                ))

        return positions

    def get_open_trades(self) -> List[Trade]:
        """Get all open trades"""
        endpoint = f"{self.config.base_url}/v3/accounts/{self.config.account_id}/openTrades"

        response = requests.get(endpoint, headers=self.config.headers)

        if response.status_code != 200:
            raise Exception(f"Failed to get trades: {response.text}")

        trades = []
        for trade_data in response.json().get("trades", []):
            trades.append(Trade(
                id=trade_data["id"],
                instrument=trade_data["instrument"],
                units=float(trade_data["currentUnits"]),
                price=float(trade_data["price"]),
                unrealized_pnl=float(trade_data.get("unrealizedPL", 0)),
                realized_pnl=float(trade_data.get("realizedPL", 0)),
                state=trade_data["state"],
                open_time=datetime.fromisoformat(trade_data["openTime"].replace("Z", "+00:00"))
            ))

        return trades

    def place_market_order(
        self,
        instrument: str,
        units: int,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Place a market order

        Args:
            instrument: Currency pair (e.g., "EUR_USD")
            units: Number of units (positive for buy, negative for sell)
            stop_loss: Stop loss price
            take_profit: Take profit price
        """
        endpoint = f"{self.config.base_url}/v3/accounts/{self.config.account_id}/orders"

        order = {
            "order": {
                "type": "MARKET",
                "instrument": instrument,
                "units": str(units),
                "timeInForce": "FOK",
                "positionFill": "DEFAULT"
            }
        }

        # Add stop loss if provided
        if stop_loss:
            order["order"]["stopLossOnFill"] = {
                "price": str(stop_loss),
                "timeInForce": "GTC"
            }

        # Add take profit if provided
        if take_profit:
            order["order"]["takeProfitOnFill"] = {
                "price": str(take_profit),
                "timeInForce": "GTC"
            }

        response = requests.post(
            endpoint,
            headers=self.config.headers,
            json=order
        )

        if response.status_code not in [200, 201]:
            raise Exception(f"Failed to place order: {response.text}")

        return response.json()

    def close_trade(self, trade_id: str) -> Dict[str, Any]:
        """Close a specific trade"""
        endpoint = f"{self.config.base_url}/v3/accounts/{self.config.account_id}/trades/{trade_id}/close"

        response = requests.put(endpoint, headers=self.config.headers)

        if response.status_code != 200:
            raise Exception(f"Failed to close trade: {response.text}")

        return response.json()

    def close_all_trades(self, instrument: Optional[str] = None) -> List[Dict[str, Any]]:
        """Close all open trades, optionally filtered by instrument"""
        trades = self.get_open_trades()
        results = []

        for trade in trades:
            if instrument is None or trade.instrument == instrument:
                try:
                    result = self.close_trade(trade.id)
                    results.append(result)
                except Exception as e:
                    print(f"Failed to close trade {trade.id}: {e}")

        return results

    def modify_trade(
        self,
        trade_id: str,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> Dict[str, Any]:
        """Modify stop loss and take profit of an existing trade"""
        endpoint = f"{self.config.base_url}/v3/accounts/{self.config.account_id}/trades/{trade_id}/orders"

        order = {}

        if stop_loss:
            order["stopLoss"] = {
                "price": str(stop_loss),
                "timeInForce": "GTC"
            }

        if take_profit:
            order["takeProfit"] = {
                "price": str(take_profit),
                "timeInForce": "GTC"
            }

        response = requests.put(
            endpoint,
            headers=self.config.headers,
            json=order
        )

        if response.status_code != 200:
            raise Exception(f"Failed to modify trade: {response.text}")

        return response.json()

    def calculate_position_size(
        self,
        account_balance: float,
        risk_percentage: float,
        stop_loss_pips: float,
        pip_value: float = 0.0001
    ) -> int:
        """
        Calculate position size based on risk management

        Args:
            account_balance: Account balance
            risk_percentage: Risk per trade (e.g., 0.01 for 1%)
            stop_loss_pips: Stop loss in pips
            pip_value: Value of 1 pip (default 0.0001 for most pairs)
        """
        risk_amount = account_balance * risk_percentage
        position_size = risk_amount / (stop_loss_pips * pip_value)
        return int(position_size)

    def calculate_stop_loss_take_profit(
        self,
        entry_price: float,
        direction: str,
        stop_loss_pips: float = 20,
        risk_reward_ratio: float = 2.0,
        pip_value: float = 0.0001
    ) -> tuple:
        """
        Calculate stop loss and take profit levels

        Args:
            entry_price: Entry price
            direction: 'BUY' or 'SELL'
            stop_loss_pips: Stop loss in pips
            risk_reward_ratio: Risk/reward ratio
            pip_value: Value of 1 pip
        """
        stop_loss_distance = stop_loss_pips * pip_value
        take_profit_distance = stop_loss_distance * risk_reward_ratio

        if direction == 'BUY':
            stop_loss = round(entry_price - stop_loss_distance, 5)
            take_profit = round(entry_price + take_profit_distance, 5)
        else:  # SELL
            stop_loss = round(entry_price + stop_loss_distance, 5)
            take_profit = round(entry_price - take_profit_distance, 5)

        return stop_loss, take_profit