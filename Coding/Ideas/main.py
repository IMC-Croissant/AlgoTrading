from typing import Dict, List, Tuple
from datamodel import OrderDepth, TradingState, Order
import numpy as np
import pandas as pd


class Trader:
    def _nadaraya_watson_estimator(
        self, data: pd.DataFrame, target: str, bandwidth: float
    ) -> np.ndarray:
        """
        Computes the Nadaraya-Watson Estimator for given data and target

        Returns:
            array of estimated target values
        """
        target_values = data[target].values
        num_instances = data.shape[0]
        estimated_target_values = np.zeros(num_instances)

        for i in range(num_instances):
            distances = np.abs(data[target] - target_values[i])
            weights = np.exp(-0.5 * (distances / bandwidth) ** 2)
            weights /= np.sum(weights)
            estimated_target_values[i] = np.dot(weights, target_values)

        return estimated_target_values

    def _fibonacci_levels_estimator(
        self, high: float, low: float, num_levels: int
    ) -> list:
        """
        Computes the Fibonacci retracement levels given high and low prices.

        Args:
            high: high price
            low: low price
            num_level: number of levels to compute

        Returns:
            list: fibonacci retracement levels
        """
        levels = []
        for level in num_levels:
            levels.append(low + (high - low) * level / (num_levels + 1))

        return levels

    def _hurst_exponential(
        self, data: pd.DataFrame, min_lag: int = 1, max_lag: int = 100
    ) -> Tuple[list, np.ndarray, list]:
        """Computes Hurst exponential estimate H following
        min and max lag values.

        Remark: H > 0.5 -> single direction, H < 0.5 -> oscillation
        otherwise H = 0.5 -> randomness
        """
        lags = np.arange(min_lag, max_lag)
        tau = [np.std(np.substract(data[lag:], price[:-lag]) for lag in lags)]
        poly = np.polyfit(np.log10(lags), np.log10(tau), 1)

        return poly, lags, tau

    def _get_acceptable_price(self, state, product, bandwidth):
        """Computes the acceptable price given a state, target
        and bandwitdth using Nadaraya-Watson estimator."""
        order_depths = state.order_depths[product]  # List[Trade]
        orders = pd.DataFrame(
            {
                "buy_value": [int(amount) for amount in order_depths.buy_orders.keys()],
                "sell_value": [
                    int(amount) for amount in order_depths.sell_orders.keys()
                ],
                "buy_volume": [
                    int(amount) for amount in order_depths.sell_orders.values()
                ],
                "sell_volume": [
                    int(amount) for amount in order_depths.sell_orders.values()
                ],
            }
        )

        vol_buy = float(orders["buy_volume"].sum())
        vol_sell = float(orders["sell_volume"].sum())

        if vol_buy > vol_sell:
            buy_val = orders["buy_value"]
            acceptable_price = float(buy_val.max() + buy_val.std())
        else:
            sell_val = orders["sell_value"]
            acceptable_price = float(sell_val.min() - sell_val.std())

        # processed_data = pd.DataFrame(
        #     diff, columns=[product]
        # )
        # estimated_target_values = self._nadaraya_watson_estimator(
        #     processed_data, product, bandwidth
        # )
        print(f"get_acceptable_price: {acceptable_price}")

        return acceptable_price

    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        """
        Only method required. It takes all buy and sell orders for all symbols as an input,
        and outputs a list of orders to be sent
        """
        # Initialize the method output dict as an empty dict
        result = {}

        # Iterate over all the keys (the available products) contained in the order dephts
        for product in state.order_depths.keys():

            # Check if the current product is the 'PEARLS' product, only then run the order logic
            if product in ("PEARLS", "BANANAS"):

                # Retrieve the Order Depth containing all the market BUY and SELL orders for PEARLS
                order_depth: OrderDepth = state.order_depths[product]
                # Initialize the list of Orders to be sent as an empty list
                orders: list[Order] = []

                # Define a fair value for the PEARLS.
                # Note that this value of 1 is just a dummy value, you should likely change it!
                bandwidth = 20 if product == "PEARLS" else 5
                acceptable_price = self._get_acceptable_price(state, product, bandwidth)

                # If statement checks if there are any SELL orders in the PEARLS market
                if len(order_depth.sell_orders) > 0:

                    # Sort all the available sell orders by their price,
                    # and select only the sell order with the lowest price
                    best_ask = min(order_depth.sell_orders.keys())
                    best_ask_volume = order_depth.sell_orders[best_ask]

                    # Check if the lowest ask (sell order) is lower than the above defined fair value
                    if best_ask < acceptable_price:

                        # In case the lowest ask is lower than our fair value,
                        # This presents an opportunity for us to buy cheaply
                        # The code below therefore sends a BUY order at the price level of the ask,
                        # with the same quantity
                        # We expect this order to trade with the sell order
                        print("BUY", str(-best_ask_volume) + "x", best_ask)
                        orders.append(Order(product, best_ask, -best_ask_volume))

                # The below code block is similar to the one above,
                # the difference is that it find the highest bid (buy order)
                # If the price of the order is higher than the fair value
                # This is an opportunity to sell at a premium
                if len(order_depth.buy_orders) != 0:
                    best_bid = max(order_depth.buy_orders.keys())
                    best_bid_volume = order_depth.buy_orders[best_bid]
                    if best_bid > acceptable_price:
                        print("SELL", str(best_bid_volume) + "x", best_bid)
                        orders.append(Order(product, best_bid, -best_bid_volume))

                # Add all the above the orders to the result dict
                result[product] = orders

                # Return the dict of orders
                # These possibly contain buy or sell orders for PEARLS
                # Depending on the logic above
        return result
