from typing import Dict, List
from datamodel import Listing, OrderDepth, Trade, TradingState, Order
from numpy import NaN, array
import pandas as pd
import math


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

def supertrend(close: List[float], high: List[float], low: List[float], period: int = 7, multiplier: float = 3.0) -> pd.DataFrame:
    df = pd.DataFrame({'Close': close, 'High': high, 'Low': low})
    ATR = period * df['High'].diff().abs().fillna(0)
    df['Basic Upper Band'] = (df['High'] + df['Low']) / 2 + multiplier * ATR
    df['Basic Lower Band'] = (df['High'] + df['Low']) / 2 - multiplier * ATR
    df['Final Upper Band'] = NaN
    df['Final Lower Band'] = NaN
    for i in range(period, len(df)):
        df.at[i, 'Final Upper Band'] = df['Basic Upper Band'][i] if df['Basic Upper Band'][i] < df['Final Upper Band'][i-1] or pd.isna(df['Final Upper Band'][i-1]) else df['Final Upper Band'][i-1]
        df.at[i, 'Final Lower Band'] = df['Basic Lower Band'][i] if df['Basic Lower Band'][i] > df['Final Lower Band'][i-1] or pd.isna(df['Final Lower Band'][

# Parkinson’s volatility uses the stock’s high and low price of the day rather than just close to close prices. It’s useful to capture large price movements during the day.
"""
The ParkinsonVolatility class takes a list of prices and a period parameter as inputs. 
It then calculates the logarithm of the highest and lowest prices in each period of length period, and stores them in a list called log_hl. 
Finally, it calculates the square root of the sum of the squares of the elements in log_hl multiplied by 1 / (len(self.prices) - self.period), 
which is the number of non-overlapping periods in the price data. This value represents the Parkinson's volatility of the price data over the specified period.
"""     
class ParkinsonVolatility:
    def __init__(self, prices: List[float], period: int):
        self.prices = prices
        self.period = period

    def calculate(self) -> float:
        log_hl = []
        for i in range(len(self.prices) - self.period):
            hl = math.log(max(self.prices[i:i+self.period])) - math.log(min(self.prices[i:i+self.period]))
            log_hl.append(hl)
        return math.sqrt(sum([x ** 2 for x in log_hl]) * (1 / (len(self.prices) - self.period)))
