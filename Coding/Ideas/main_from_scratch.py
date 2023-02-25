from typing import Dict, List
from datamodel import OrderDepth, TradingState, Order
from pandas import DataFrame


class Trader:
    _history = DataFrame({
        'PEARLS': 9980,
        'BANANAS': 4500,
            })

    def _get_acceptable_price(self, state: TradingState, product: str):
        """Computes acceptable price from historical data. """
        history_product = self._history[product]
        # compute rolling mean of size 20 if possible
        print("state timestamp ", state.timestamp)
        if state.timestamp > 2000:
            history_product.rolling(window=10).mean()
        else:
            acceptable_price = history_product.mean()

        return acceptable_price

    def _process_new_data(self, state: TradingState) -> None:
        """Adds new data point to historical data."""
        # initialize values at the beginning
        # own_trades still not populated
        if ("PEARLS" not in state.own_trades.keys() or "BANANAS" not in state.own_trades.keys()):
            self._history.append({
                "PEARLS": 9980,
                "BANANAS": 4890,
                },
                ignore_index=True)
        else:
            # own_trades[product] is a tuple (product, << SUBMISSION, price, quantity)
            own_trades = state.own_trades
            self._history.append({
                "PEARLS": own_trades["PEARLS"][2],
                "BANANAS": own_trades["BANANAS"][2]
                })

    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        """
        Only method required. It takes all buy and sell orders for all symbols as an input,
        and outputs a list of orders to be sent
        """
        # get history
        # self._history.append(state.timestamp)
        # print("history size ", len(self._history))
        # Initialize the method output dict as an empty dict
        result = {}

        print("current state own orders ", state.own_trades)
        print("current state observation ", state.observations)
        # process new data
        self._process_new_data(state)

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
                # acceptable_price = 4900 if product == 'PEARLS' else 9980
                acceptable_price = self._get_acceptable_price(state, product)
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
                # the difference is that it finds the highest bid (buy order)
                # If the price of the order is higher than the fair value
                # This is an opportunity to sell at a premium
                if len(order_depth.buy_orders) != 0:
                    best_bid = max(order_depth.buy_orders.keys())
                    best_bid_volume = order_depth.buy_orders[best_bid]
                    if best_bid > acceptable_price:
                        print("SELL", str(best_bid_volume) + "x", best_bid)
                        orders.append(Order(product, best_bid, -best_bid_volume))

                # Add all the above orders to the result dict
                result[product] = orders

                # Return the dict of orders
                # These possibly contain buy or sell orders for PEARLS
                # Depending on the logic above
        return result
