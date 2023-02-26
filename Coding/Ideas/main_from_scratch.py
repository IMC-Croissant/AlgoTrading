from typing import Dict, List
from datamodel import OrderDepth, TradingState, Order
from pandas import DataFrame
import numpy as np


class Trader:
    _history = DataFrame({
        'PEARLS': [10000],
        'BANANAS': [10000],
            })

    def _hurst_exponential(self, product: str, timestamp: int, min_lag: int, max_lag: int) -> bool:
        """Computes Hurst exponential estimate H following
        min and max lag values.

        Remark: H > 0.5 -> trendy, H < 0.5 -> reversal
        otherwise H = 0.5 -> randomness
        """
        if min_lag < 1:
            raise Exception("min_lag must be >= 1")
        # Added condition to make sure hurst exponent can be computed
        if timestamp < (max_lag + 1) * 100:
            return True

        data = self._history[product].values
        lags = np.arange(min_lag, max_lag + 1)
        tau = [np.std(np.subtract(data[lag:], data[:-lag]))
            for lag in lags]
        poly = np.polyfit(np.log10(lags), np.log10(tau), 1)
        trendy = True if poly[0] > 0.5 else False
        print("Product {} is trendy {}".format(product, trendy))

        return trendy

    def _get_acceptable_price(self, state: TradingState, product: str) -> tuple:
        """Computes acceptable price from historical data. """
        history_product = self._history[product]
        # print("history_product \n", history_product)
        # compute rolling mean of size 20 if possible
        # print("state timestamp ", state.timestamp)
        if state.timestamp > 2000:
            history_rolling = history_product.rolling(window=8)
            means = history_rolling.mean()
            stds = history_rolling.std()
            # fill na's 
            means = means.fillna(history_product.mean())
            stds = stds.fillna(0)
            # if means.isnull().values.any():
            #     acceptable_price = history_product.iloc[0]
            #     std = 0.0
            acceptable_price = means.mean()
            std = stds.mean()

        else:
            acceptable_price = history_product.mean()
            std = 0.0

        print("computed acceptable price {} std {} for {}".format(
            acceptable_price, std, product))

        return acceptable_price, std

    def _process_new_data(self, state: TradingState) -> None:
        """Adds new data point to historical data."""
        # initialize values at the beginning
        # own_trades still not populated
        if ("PEARLS" not in state.own_trades.keys() or "BANANAS" not in state.own_trades.keys()):
            pass
        else:
            # own_trades[product] is a Trade object
            # it has attributed (symbol, price, quantity, buyer, seller)
            own_trades = state.own_trades
            # getting average price per share
            avg_pearls = sum([trade.price * trade.quantity for trade in own_trades["PEARLS"]])
            avg_pearls /= sum([trade.quantity for trade in own_trades["PEARLS"]])
            avg_bananas = sum([trade.price * trade.quantity for trade in own_trades["BANANAS"]])
            avg_bananas /= sum([trade.quantity for trade in own_trades["BANANAS"]])

            # print("avg_bananas, ", avg_bananas)
            # print("avg_pearls, ", avg_pearls)
            self._history = self._history.append({
                "PEARLS": avg_pearls,
                "BANANAS": avg_bananas,
                },
                ignore_index=True)

    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        """
        Only method required. It takes all buy and sell orders for all symbols as an input,
        and outputs a list of orders to be sent
        """
        # Initialize the method output dict as an empty dict
        result = {}

        print("current state own orders ", state.own_trades)
        # print("current state observation ", state.observations)
        # process new data
        self._process_new_data(state)
        # print("current data ", self._history)

        # Iterate over all the keys (the available products) contained in the order depths
        for product in state.order_depths.keys():

            # Check if the current product is the 'PEARLS' product, only then run the order logic
            if product in ('PEARLS', 'BANANAS'):

                # Retrieve the Order Depth containing all the market BUY and SELL orders for PEARLS
                order_depth: OrderDepth = state.order_depths[product]

                # Initialize the list of Orders to be sent as an empty list
                orders: list[Order] = []

                # Define a fair value for the PEARLS.
                # Note that this value of 1 is just a dummy value, you should likely change it!
                acceptable_price, std = self._get_acceptable_price(state, product)
                # Identify trendy market with Hurst exponent
                trendy = self._hurst_exponential(product, state.timestamp, 1, 15)
                # If statement checks if there are any SELL orders in the PEARLS market
                if len(order_depth.sell_orders) > 0:

                    # Sort all the available sell orders by their price,
                    # and select only the sell order with the lowest price
                    best_ask = min(order_depth.sell_orders.keys())
                    best_ask_volume = order_depth.sell_orders[best_ask]

                    # Check if the lowest ask (sell order) is lower than the above defined fair value
                    if best_ask < acceptable_price - 2*std:

                        # In case the lowest ask is lower than our fair value,
                        # This presents an opportunity for us to buy cheaply
                        # The code below therefore sends a BUY order at the price level of the ask,
                        # with the same quantity
                        # We expect this order to trade with the sell order
                        print("BUY", str(-best_ask_volume) + "x", best_ask)
                        orders.append(Order(product, best_ask, -best_ask_volume))

                # The below code block is similar to the one above,
                # the difference is that it finds the highest bid (buy order)
                # If the price of the order is higher than the fair value
                # This is an opportunity to sell at a premium
                if len(order_depth.buy_orders) != 0:
                    best_bid = max(order_depth.buy_orders.keys())
                    best_bid_volume = order_depth.buy_orders[best_bid]
                    if best_bid > acceptable_price + 2*std and not trendy:
                        print("SELL", str(best_bid_volume) + "x", best_bid)
                        orders.append(Order(product, best_bid, -best_bid_volume))

                # Add all the above orders to the result dict
                result[product] = orders

                # Return the dict of orders
                # These possibly contain buy or sell orders for PEARLS
                # Depending on the logic above
        return result
