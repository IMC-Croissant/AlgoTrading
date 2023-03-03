from typing import Dict, List
from datamodel import OrderDepth, TradingState, Order
from pandas import DataFrame
import math
import pandas as pd
import numpy as np


class Trader:
    _history = pd.DataFrame([[10, 144]],
            columns=['PEARLS', 'BANANAS'],
            index=[0])

    def _get_sma_indicator(self, state: TradingState, product: str) -> bool:
        """Computes SMA20 and SMA50 and defines a bullish indicator."""
        history_product = self._history[product]

        bullish = False

        if state.timestamp > 51 * 100:
            sma_20 = history_product.rolling(window=20).mean()[state.timestamp]
            sma_50 = history_product.rolling(window=50).mean()[state.timestamp]

            if sma_50 < sma_20:
                bullish = True # bullish market
            else:
                bullish = False

        print("SMA indicator is bullish: ", bullish)

        return bullish

    def _hurst_exponential(self, product: str, timestamp: int, min_lag: int, max_lag: int) -> bool:
        """Computes Hurst exponential estimate H following
        min and max lag values.

        Remark: H > 0.5 -> bullish, H < 0.5 -> reversal
        otherwise H = 0.5 -> randomness
        """
        if min_lag < 1:
            raise Exception("min_lag must be >= 1")
        # Added condition to make sure hurst exponent can be computed
        if timestamp < (max_lag + 1) * 100:
            return False

        data = self._history[product].values
        lags = np.arange(min_lag, max_lag + 1)
        tau = [np.std(np.subtract(data[lag:], data[:-lag]))
            for lag in lags]
        poly = np.polyfit(np.log10(lags), np.log10(tau), 1)
        bullish = True if poly[0] > 0.5 else False
        print("Product {} is bullish (Hurst) {}".format(product, bullish))

        return bullish

    def _get_acceptable_price(self, state: TradingState, product: str, bullish: bool) -> tuple:
        """Computes acceptable price from historical data. """
        # history_product = self._history[product]
        # print("history_product \n", history_product)
        # print("state timestamp ", state.timestamp)
        min_ask = min(state.order_depths[product].sell_orders)
        max_bid = max(state.order_depths[product].buy_orders)
        mid_price = (min_ask + max_bid) / 2
        spread = min_ask - max_bid

        # we should adapt based on product
        if spread > 6 and bullish:
            acceptable_bid = max_bid
            acceptable_ask = min_ask
        elif spread > 6 and not bullish:
            acceptable_bid = max_bid
            acceptable_ask = min_ask
        else:
            acceptable_bid = max_bid
            acceptable_ask = min_ask

        acceptable_bid = math.ceil(acceptable_bid)
        acceptable_ask = math.floor(acceptable_ask)

        print("acceptable bid {} ask {} product {}".format(
            acceptable_bid, acceptable_ask, product))

        return acceptable_bid, acceptable_ask, mid_price

    def _process_new_data(self, state: TradingState) -> None:
        """Adds new data point to historical data."""
        # initialize values at the beginning
        # own_trades still not populated
        if not bool(state.order_depths):
            return

        mid_prices = []
        for product in state.order_depths.keys():
            min_ask = min(state.order_depths[product].sell_orders)
            max_bid = max(state.order_depths[product].buy_orders)
            mid_price = (min_ask + max_bid) / 2
            mid_prices.append(mid_price)

        products = [product for product in state.order_depths.keys()]
        temp_dataframe = pd.DataFrame([mid_prices],
            columns=products,
            index=[state.timestamp])

        self._history = pd.concat([self._history, temp_dataframe])


    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        """
        Only method required. It takes all buy and sell orders for all symbols as an input,
        and outputs a list of orders to be sent
        """
        result = {}

        # print("current state own orders ", state.own_trades)
        # print("current state observation ", state.observations)
        self._process_new_data(state)
        # print("current data ", self._history)

        for product in state.order_depths.keys():
            # order_depth: OrderDepth = state.order_depths[product]
            orders: list[Order] = []

            # bullish = self._hurst_exponential(product, state.timestamp, 1, 40)
            bullish = self._get_sma_indicator(state, product)
            acceptable_bid, acceptable_ask, mid_price = self._get_acceptable_price(
                state, product, bullish)

            current_position = 0
            if bool(state.position):
                if product in state.position.keys():
                    current_position = state.position[product]

            max_long_position = 20 - current_position
            max_short_position = -20 - current_position

            buy_quantity = min(15, max_long_position)
            sell_quantity = max(-15, max_short_position)

            if product == "PEARLS":
                # bid ask for only large spreads
                if mid_price > 9997:
                    orders.append(Order(product, acceptable_ask, sell_quantity))
                if mid_price < 10003:
                    orders.append(Order(product, acceptable_bid, buy_quantity))
            if product == "BANANAS":
                orders.append(Order(product, acceptable_ask, sell_quantity))
                orders.append(Order(product, acceptable_bid, buy_quantity))

            result[product] = orders

        return result
