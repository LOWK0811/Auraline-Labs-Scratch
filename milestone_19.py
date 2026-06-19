# ======================================================================
# SECTION 1: IMPORTS
# ======================================================================
import os
import logging
import json
from datetime import datetime
from src.data_handler import get_price_data

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


# ======================================================================
# SECTION 2: LOCAL PAPER BROKER
# ======================================================================
class PaperBroker:
    """
    A simulated broker that mimics a real trading API.
    Tracks cash, positions, and order history locally.
    State is saved to a JSON file so it persists between runs.
    """

    def __init__(self, starting_cash=100000, state_file="data/paper_account.json"):
        self.state_file = state_file
        self._load_or_init(starting_cash)

    def _load_or_init(self, starting_cash):
        if os.path.exists(self.state_file):
            with open(self.state_file, "r") as f:
                state = json.load(f)
            self.cash      = state["cash"]
            self.positions = state["positions"]
            self.orders    = state["orders"]
            logger.info("Loaded existing paper account state")
        else:
            self.cash      = starting_cash
            self.positions = {}   # ticker -> {"qty": int, "avg_price": float}
            self.orders    = []
            self._save()
            logger.info(f"New paper account created with ${starting_cash:,.2f}")

    def _save(self):
        os.makedirs("data", exist_ok=True)
        with open(self.state_file, "w") as f:
            json.dump({
                "cash":      self.cash,
                "positions": self.positions,
                "orders":    self.orders
            }, f, indent=2)

    # ======================================================================
    # SECTION 3: ACCOUNT INFO
    # ======================================================================
    def get_account(self, current_prices=None):
        """Returns account summary."""
        portfolio_value = self.cash
        if current_prices:
            for ticker, pos in self.positions.items():
                price = current_prices.get(ticker, pos["avg_price"])
                portfolio_value += pos["qty"] * price

        logger.info(f"Cash:            ${self.cash:,.2f}")
        logger.info(f"Portfolio value: ${portfolio_value:,.2f}")
        logger.info(f"Open positions:  {len(self.positions)}")
        return {"cash": self.cash, "portfolio_value": portfolio_value}

    # ======================================================================
    # SECTION 4: PLACE ORDERS
    # ======================================================================
    def buy(self, ticker, qty, price):
        """Simulates a market buy order."""
        cost = qty * price * 1.001   # include friction
        if cost > self.cash:
            logger.warning(f"Insufficient cash to buy {qty} {ticker} "
                          f"@ ${price:.2f} (need ${cost:.2f}, have ${self.cash:.2f})")
            return None

        self.cash -= cost

        if ticker in self.positions:
            existing    = self.positions[ticker]
            total_qty   = existing["qty"] + qty
            avg_price   = ((existing["qty"] * existing["avg_price"]) +
                          (qty * price)) / total_qty
            self.positions[ticker] = {"qty": total_qty, "avg_price": avg_price}
        else:
            self.positions[ticker] = {"qty": qty, "avg_price": price}

        order = {
            "id":        len(self.orders) + 1,
            "timestamp": datetime.now().isoformat(),
            "ticker":    ticker,
            "side":      "buy",
            "qty":       qty,
            "price":     price,
            "cost":      cost
        }
        self.orders.append(order)
        self._save()
        logger.info(f"BUY  {qty} {ticker} @ ${price:.2f} | "
                   f"Cost: ${cost:.2f} | Cash left: ${self.cash:,.2f}")
        return order

    def sell(self, ticker, qty, price):
        """Simulates a market sell order."""
        if ticker not in self.positions:
            logger.warning(f"No position in {ticker} to sell")
            return None
        if self.positions[ticker]["qty"] < qty:
            logger.warning(f"Not enough shares to sell "
                          f"({self.positions[ticker]['qty']} held, {qty} requested)")
            return None

        proceeds = qty * price * 0.999
        self.cash += proceeds
        self.positions[ticker]["qty"] -= qty

        if self.positions[ticker]["qty"] == 0:
            del self.positions[ticker]

        order = {
            "id":        len(self.orders) + 1,
            "timestamp": datetime.now().isoformat(),
            "ticker":    ticker,
            "side":      "sell",
            "qty":       qty,
            "price":     price,
            "proceeds":  proceeds
        }
        self.orders.append(order)
        self._save()
        logger.info(f"SELL {qty} {ticker} @ ${price:.2f} | "
                   f"Proceeds: ${proceeds:.2f} | Cash: ${self.cash:,.2f}")
        return order

    # ======================================================================
    # SECTION 5: VIEW POSITIONS AND HISTORY
    # ======================================================================
    def get_positions(self, current_prices=None):
        """Prints all open positions with unrealized P&L."""
        if not self.positions:
            logger.info("No open positions")
            return

        for ticker, pos in self.positions.items():
            current_price = (current_prices or {}).get(ticker, pos["avg_price"])
            unrealized_pnl = (current_price - pos["avg_price"]) * pos["qty"]
            logger.info(f"{ticker}: {pos['qty']} shares | "
                       f"Avg entry: ${pos['avg_price']:.2f} | "
                       f"Current: ${current_price:.2f} | "
                       f"Unrealized P&L: ${unrealized_pnl:+.2f}")

    def get_order_history(self, last_n=5):
        """Prints the last N orders."""
        recent = self.orders[-last_n:]
        for order in recent:
            logger.info(f"Order #{order['id']}: {order['side'].upper()} "
                       f"{order['qty']} {order['ticker']} @ ${order['price']:.2f} "
                       f"on {order['timestamp'][:10]}")


# ======================================================================
# SECTION 6: TEST THE PAPER BROKER
# ======================================================================
broker = PaperBroker(starting_cash=100000)

data = get_price_data("AAPL", "2021-01-01", "2026-06-01")
latest_price = float(data["Close"].iloc[-1])
logger.info(f"Latest AAPL close price: ${latest_price:.2f}")

logger.info("\n=== Account before trades ===")
broker.get_account({"AAPL": latest_price})

logger.info("\n=== Placing test orders ===")
broker.buy("AAPL", 10, latest_price)
broker.buy("AAPL", 5,  latest_price)

logger.info("\n=== Current positions ===")
broker.get_positions({"AAPL": latest_price})

logger.info("\n=== Selling half the position ===")
broker.sell("AAPL", 7, latest_price)

logger.info("\n=== Account after trades ===")
broker.get_account({"AAPL": latest_price})

logger.info("\n=== Order history ===")
broker.get_order_history()