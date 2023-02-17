from typing import List, Dict
from collections import defaultdict
from datamodel import Trade

def calculate_vwap(trades: List[Trade]) -> Dict[str, float]:
    vwap_by_product = defaultdict(float)
    volume_by_product = defaultdict(int)
    for trade in trades:
        vwap_by_product[trade.symbol] += trade.price * trade.quantity
        volume_by_product[trade.symbol] += trade.quantity
    for product in vwap_by_product:
        vwap_by_product[product] /= volume_by_product[product]
    return dict(vwap_by_product)

from typing import List
from datamodel import Trade

def calculate_twap(trades: List[Trade], start_time: int, end_time: int) -> float:
    total_value = 0.0
    total_volume = 0.0
    last_trade_time = start_time
    for trade in trades:
        # If the trade occurred outside the specified time range, skip it
        if trade.timestamp < start_time or trade.timestamp > end_time:
            continue
        
        # Calculate the time interval since the last trade
        time_interval = trade.timestamp - last_trade_time
        
        # Add the value of this trade to the running total
        total_value += trade.price * trade.quantity
        
        # Add the volume of this trade to the running total
        total_volume += trade.quantity
        
        # Update the last trade time
        last_trade_time = trade.timestamp
    
    # Calculate the total time interval
    total_time = end_time - start_time
    
    # Calculate the TWAP
    if total_volume == 0 or total_time == 0:
        twap = 0
    else:
        twap = total_value / total_volume
    
    return twap

