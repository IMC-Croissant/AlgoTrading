from typing import Dict, List
from datamodel import Listing, OrderDepth, Trade, TradingState, Order

def vwap(trades: List[Trade], volume: int) -> float:
    trade_sum = 0
    volume_remaining = volume
    for trade in trades:
        if trade.quantity <= volume_remaining:
            trade_sum += trade.price * trade.quantity
            volume_remaining -= trade.quantity
        else:
            trade_sum += trade.price * volume_remaining
            break
    return trade_sum / volume

def twap(trades: List[Trade], volume: int, time_delta: int) -> float:
    trade_sum = 0
    volume_remaining = volume
    time_elapsed = 0
    for trade in trades:
        if trade.quantity <= volume_remaining:
            trade_sum += trade.price * trade.quantity
            volume_remaining -= trade.quantity
            time_elapsed += (trade.timestamp - time_elapsed) * trade.quantity / volume
        else:
            trade_sum += trade.price * volume_remaining
            time_elapsed += (trade.timestamp - time_elapsed) * volume_remaining / volume
            break
    return trade_sum / volume

def calculate_bollinger_bands(state: TradingState, symbol: str, window: int, num_std_dev: int) -> List[Tuple[float, float, float]]:
    """
    Calculates the Bollinger Bands for the given symbol over the specified window size.
    The window size is the number of minutes to use for the moving average.
    The number of standard deviations to use is given by num_std_dev.
    Returns a list of tuples, where each tuple contains the moving average, upper band, and lower band for a given timestamp.
    """
    order_depth = state.order_depths[symbol]
    trades = state.market_trades[symbol]
    prices = [trade.price for trade in trades]
    if len(prices) < window:
        raise ValueError(f"Not enough data for symbol {symbol} over the specified window size.")
    ma_values = []
    std_dev_values = []
    bollinger_bands = []
    for i in range(window, len(prices)):
        ma = sum(prices[i-window:i]) / window
        std_dev = (sum([(p - ma)**2 for p in prices[i-window:i]]) / window)**0.5
        upper_band = ma + num_std_dev * std_dev
        lower_band = ma - num_std_dev * std_dev
        ma_values.append(ma)
        std_dev_values.append(std_dev)
        bollinger_bands.append((ma, upper_band, lower_band))
    return bollinger_bands

