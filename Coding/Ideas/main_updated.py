from typing import Dict, List
from datamodel import OrderDepth, TradingState, Order
from pandas import DataFrame
import math
import pandas as pd
import numpy as np


class Trader:
    _history = pd.DataFrame([[10000, 4950]],
            columns=['PEARLS', 'BANANAS'],
            index=[0])

    def _get_sma_values_and_indicator(self, state: TradingState, product: str) -> bool:
        """Computes SMA5, SMA15, SMA40 and SMA90 with respective bands values"""
        history_product = self._history[product]
        current_mid_price = history_product[state.timestamp]
        # std_sma_20 = history_product.rolling(window=20).std()[state.timestamp]
        # sma_20 = history_product.rolling(window=20).mean()[state.timestamp]

        bullish = -1
        sma_5, sma_15, sma_40, sma_90 = -1, -1, -1, -1

        if state.timestamp > 5 * 100:
            sma_5 = history_product.rolling(window=5).mean()[state.timestamp]
        if state.timestamp > 15 * 100:
            sma_15 = history_product.rolling(window=15).mean()[state.timestamp]
        if state.timestamp > 40 * 100:
            sma_40 = history_product.rolling(window=40).mean()[state.timestamp]
        if state.timestamp > 90 * 100:
            sma_90 = history_product.rolling(window=90).mean()[state.timestamp]

        values = [sma_5, sma_15, sma_40, sma_90]

        if product == "BANANAS":
            if sma_40 < sma_15:
                bullish = True
            elif sma_90 < sma_15 and sma_40 < sma_15:
                bullish = True
            else:
                bullish = False

        if product == "PEARLS":
            pass

        print("SMA indicator is bullish {} for {}".format(bullish, product))
        return values, bullish

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

    def _get_acceptable_quantity(
            self,
            state: TradingState,
            product: int,
            bullish: bool,
            position_limit: int = 20) -> tuple:
        """Computes acceptable quantity (volume) from position. """
        current_volume = 0

        if bool(state.position):
            if product in state.position.keys():
                current_volume = state.position[product]

        max_long_position = 20 - current_volume
        max_short_position = -20 - current_volume

        buy_volume = min(20, max_long_position)
        sell_volume = max(-20, max_short_position)

        # if market is trendy => adjust volumes
        if product == "BANANAS":
            if isinstance(bullish, bool) and bullish:
                sell_volume -= 1
            elif isinstance(bullish, bool) and not bullish:
                buy_volume += 1

            if current_volume >= 10 and isinstance(bullish, bool) and not bullish:
                sell_volume += 1
            if current_volume <= -10 and isinstance(bullish, bool) and bullish:
                buy_volume += 1

        # TODO Add case for PEARLS
        buy_volume = min(20, max_long_position)
        sell_volume = max(-20, max_short_position)

        print("acceptable buy vol {} sell vol {} product {}".format(
            buy_volume, sell_volume, product))

        return buy_volume, sell_volume

    def _get_acceptable_price(
            self,
            state: TradingState,
            product: str,
            fair_prices: tuple,
            bullish: bool) -> tuple:
        """Computes acceptable price from historical data. """
        # history_product = self._history[product]
        # print("history_product \n", history_product)
        # print("state timestamp ", state.timestamp)

        acceptable_bid = 0
        acceptable_ask = 1000000

        asks = sorted(state.order_depths[product].sell_orders)
        bids = sorted(state.order_depths[product].buy_orders)

        l1_bid = bids[-1]
        l1_ask = asks[0]

        l2_bid = 0
        l2_ask = 1000000

        l3_bid = 0
        l3_ask = 1000000

        # If crossing the books we have the check over each level
        if len(bids) >= 3:
            l3_bid = bids[-3]
            l2_bid = bids[-2]
        elif len(bids) >= 2:
            l2_bid = bids[-2]

        if  len(asks) >= 3:
            l3_ask = asks[2]
            l2_ask = asks[1]
        elif len(asks) >= 2:
            l2_ask = asks[1]

        spread = l1_ask - l1_bid

        fair_value = fair_prices[0] # get sma_5

        if product == "PEARLS":
            # get sma_90
            fair_value = fair_prices[-1] if fair_prices[-1] > -1 else 10000
            # still not crossing the books
            if spread > 3:
                if isinstance(bullish, bool) and bullish:
                    acceptable_bid = l1_bid + 1
                    acceptable_ask = l1_ask
                elif isinstance(bullish, bool) and not bullish:
                    acceptable_bid = l1_bid
                    acceptable_ask = l1_ask - 1
                else:
                    acceptable_bid = l1_bid
                    acceptable_ask = l1_ask
            elif spread <= 3:
                # crossing the book
                if l3_bid > fair_value:
                    acceptable_ask = l3_bid
                elif l2_bid > fair_value:
                    acceptable_ask = l2_bid
                elif l1_bid > fair_value:
                    acceptable_ask = l1_bid

                if l3_ask < fair_value:
                    acceptable_bid = l3_ask
                elif l2_ask < fair_value:
                    acceptable_bid = l2_ask
                elif l1_ask < fair_value:
                    acceptable_bid = l1_ask

        if product == "BANANAS":
            if spread > 2:
                if isinstance(bullish, bool) and bullish:
                    acceptable_bid = l1_bid + 1
                    acceptable_ask = l1_ask
                elif isinstance(bullish, bool) and not bullish:
                    acceptable_bid = l1_bid
                    acceptable_ask = l1_ask - 1
                else:
                    acceptable_bid = l1_bid
                    acceptable_ask = l1_ask - 1
            elif spread <= 2:
                # crossing the book
                if l3_bid > fair_value:
                    acceptable_ask = l3_bid
                elif l2_bid > fair_value:
                    acceptable_ask = l2_bid
                elif l1_bid > fair_value:
                    acceptable_ask = l1_bid

                if l3_ask < fair_value:
                    acceptable_bid = l3_ask
                elif l2_ask < fair_value:
                    acceptable_bid = l2_ask
                elif l1_ask < fair_value:
                    acceptable_bid = l1_ask


        acceptable_bid = math.ceil(acceptable_bid)
        acceptable_ask = math.floor(acceptable_ask)

        print("acceptable bid {} ask {} product {}".format(
            acceptable_bid, acceptable_ask, product))

        return acceptable_bid, acceptable_ask

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

            fair_prices, bullish = self._get_sma_values_and_indicator(
                    state, product)

            acceptable_bid, acceptable_ask = self._get_acceptable_price(
                state, product, fair_prices, bullish)
            # get quantities to place orders
            buy_quantity, sell_quantity = self._get_acceptable_quantity(
                    state, product, bullish)
            # place orders
            orders.append(Order(product, acceptable_bid, buy_quantity))
            orders.append(Order(product, acceptable_ask, sell_quantity))

            result[product] = orders

        return result
