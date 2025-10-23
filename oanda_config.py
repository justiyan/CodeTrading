import os
from typing import Optional
from dataclasses import dataclass

@dataclass
class OandaConfig:
    """Configuration for Oanda API connection"""
    account_id: str
    api_token: str
    environment: str = "practice"  # "practice" or "live"

    @property
    def base_url(self) -> str:
        if self.environment == "practice":
            return "https://api-fxpractice.oanda.com"
        elif self.environment == "live":
            return "https://api-fxtrade.oanda.com"
        else:
            raise ValueError(f"Invalid environment: {self.environment}")

    @property
    def stream_url(self) -> str:
        if self.environment == "practice":
            return "https://stream-fxpractice.oanda.com"
        elif self.environment == "live":
            return "https://stream-fxtrade.oanda.com"
        else:
            raise ValueError(f"Invalid environment: {self.environment}")

    @property
    def headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        }

    @classmethod
    def from_env(cls) -> 'OandaConfig':
        """Create config from environment variables"""
        account_id = os.getenv("OANDA_ACCOUNT_ID")
        api_token = os.getenv("OANDA_API_TOKEN")
        environment = os.getenv("OANDA_ENVIRONMENT", "practice")

        if not account_id or not api_token:
            raise ValueError(
                "OANDA_ACCOUNT_ID and OANDA_API_TOKEN environment variables must be set"
            )

        return cls(
            account_id=account_id,
            api_token=api_token,
            environment=environment
        )