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
def parkinson_volatility(data: List[float], window_size: int = 20) -> List[float]:
    """
    Calculate Parkinson's volatility for a given data using a rolling window of window_size
    
    Parameters:
    data (List[float]): A list of float numbers representing the price data
    window_size (int): An integer representing the size of the rolling window. Default value is 20.
    
    Returns:
    A list of float numbers representing the calculated Parkinson's volatility
    
    The function loops through the price data, calculates the logarithm of the ratio between each price and the previous price, 
    and sums the squared logarithms within the rolling window. 
    The sum is then divided by the size of the window times the logarithm of 2, 
    and the result is appended to the parkinson_vol list. T
    he function returns the list of calculated Parkinson's volatilities.
    """
    half_window_size = int(window_size / 2)
    parkinson_vol = []
    for i in range(len(data)):
        if i < half_window_size:
            parkinson_vol.append(0.0)
            continue
        if i + half_window_size >= len(data):
            parkinson_vol.append(0.0)
            break
        log_sum = 0.0
        for j in range(i - half_window_size, i + half_window_size):
            log_sum += log(data[j] / data[j-1]) ** 2
        parkinson_vol.append(log_sum / (2 * half_window_size * log(2)))
    return parkinson_vol
