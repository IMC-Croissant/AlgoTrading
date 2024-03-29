from typing import Dict, List
from datamodel import OrderDepth, TradingState, Order
from pandas import DataFrame
import math
import pandas as pd
import numpy as np


class Trader:
    _history = pd.DataFrame(columns=['PEARLS', 'BANANAS'])  # gets replaced in first iteration
    _current_inventory = pd.DataFrame(columns=['product',
                                               'timestamp',
                                               'position',
                                               'avg_cost'])  # gets replaced in first interation
    _total_profit = 0

    def _update_inventory(self, state: TradingState, product: str) -> None:
        """
        Calculates profit and updates inventory for a single position using a trading state's own_trades
        """
        # Build initial dataframe at timestamp 0 before any trades have gone through
        if len(self._current_inventory) == 0:
            for product in ['PEARLS', 'BANANAS']:
                self._current_inventory = pd.concat([self._current_inventory,
                                                     pd.DataFrame([[product, 0, 0, 0]],
                                                                  columns=['product', 'timestamp',
                                                                           'position', 'avg_cost'])])

        # set position and cost to previous values
        most_recent_row = self._current_inventory[self._current_inventory['product'] == product].iloc[-1, :]
        position = most_recent_row['position']
        cost = most_recent_row['avg_cost']

        # iterate through every own trade in the state
        print(f"product {product} own trades")
        if product in state.own_trades:
            for trade in state.own_trades[product]:
                if trade.timestamp == state.timestamp - 100:
                    print(trade)
                    # If we are the buyer
                    if trade.buyer == 'SUBMISSION':
                        if position >= 0:
                            # If we are long, this indicates that we are adding more to our long position
                            cost = (cost * position + trade.price * trade.quantity) / (position + trade.quantity)
                            position += trade.quantity
                        elif position < 0 and position + trade.quantity <= 0:
                            # If we are short, this and we buy more, but not enough to create a long position.
                            self._total_profit += ((cost * -1) - trade.price) * trade.quantity
                            if position + trade.quantity == 0:
                                cost = 0
                            position += trade.quantity
                        elif position < 0 and position + trade.quantity > 0:
                            # If we are short, this and we buy enough to create a long position.
                            quantity_to_close = -position
                            self._total_profit += ((cost * -1) - trade.price) * quantity_to_close
                            cost = trade.price
                            position = trade.quantity - quantity_to_close
                    # if we are the seller
                    elif trade.seller == 'SUBMISSION':
                        if position <= 0:
                            # If we are short, this indicates that we are adding more to our short position
                            cost = ((cost * position) + (trade.price * trade.quantity)) / (position - trade.quantity)
                            position -= trade.quantity
                        elif position > 0 and position - trade.quantity >= 0:
                            self._total_profit += ((trade.price - cost) * trade.quantity)
                            if position - trade.quantity == 0:
                                cost = 0
                            position -= trade.quantity
                        elif position > 0 and position - trade.quantity < 0:
                            quantity_to_close = position
                            self._total_profit += (trade.price - cost) * quantity_to_close
                            cost = -trade.price
                            position = (trade.quantity - quantity_to_close) * -1
        else:
            print(f'No own trades for product {product}')
        print(f'Profit after previous trades for product {product} is {self._total_profit}')
        print(f'Position after previous trades for product {product} is {position}')

        self._current_inventory = pd.concat([self._current_inventory,
                                             pd.DataFrame([[product, state.timestamp, position, cost]],
                                                          columns=['product', 'timestamp',
                                                                   'position', 'avg_cost'])])

    def _get_ewm_values_and_indicator(self, state: TradingState, product: str) -> bool:
        """Computes EWM5, EWM12, EWM26, MACD and signaling."""
        history_product = self._history[product]
        current_mid_price = history_product[state.timestamp]

        bullish = -1
        ewm_5, ewm_26, signal, macd = -1, -1, -1, -1
        sma_90 = -1

        ewm_5 = history_product.ewm(span=5, adjust=False).mean()[state.timestamp]
        std_5 = history_product.ewm(span=5, adjust=False).std()[state.timestamp]
        if state.timestamp > 26 * 100:
            span_12 = history_product.ewm(span=12, adjust=False).mean()
            span_26 = history_product.ewm(span=26, adjust=False).mean()
            macd_series = span_12 - span_26
            span_9_macd = macd_series.ewm(span=9, adjust=False).mean()

            macd = macd_series[state.timestamp]
            signal = span_9_macd[state.timestamp]

        if state.timestamp > 90 * 100:
            sma_90 = history_product.ewm(span=90).mean()[state.timestamp]

        values = [ewm_5, macd, signal, sma_90]

        if product == "BANANAS":
            if signal < macd:  # bullish
                bullish = True
            elif signal > macd:
                bullish = False

        if product == "PEARLS":
            pass

        # print("MACD indicator is bullish {} for {}".format(bullish, product))
        return values, bullish

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

        # print("acceptable buy vol {} sell vol {} product {}".format(
        #    buy_volume, sell_volume, product))

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

        if len(asks) >= 3:
            l3_ask = asks[2]
            l2_ask = asks[1]
        elif len(asks) >= 2:
            l2_ask = asks[1]

        spread = l1_ask - l1_bid

        fair_value = fair_prices[0]
        cross_volume = 0
        if product == "PEARLS":
            # get sma_90
            fair_value = fair_prices[-1] if fair_prices[-1] > -1 else 10000
            # still not crossing the books
            # if spread > 3:
            #    if isinstance(bullish, bool) and bullish:
            #        acceptable_bid = l1_bid + 1
            #        acceptable_ask = l1_ask
            #    elif isinstance(bullish, bool) and not bullish:
            #        acceptable_bid = l1_bid
            #        acceptable_ask = l1_ask - 1
            #    else:
            #        acceptable_bid = l1_bid
            #        acceptable_ask = l1_ask
            # elif spread <= 3:
            # crossing the book

            if l3_bid > fair_value:
                acceptable_ask = l3_bid
                cross_volume += state.order_depths[product].buy_orders[l3_bid]
                cross_volume += state.order_depths[product].buy_orders[l2_bid]
                cross_volume += state.order_depths[product].buy_orders[l1_bid]
            elif l2_bid > fair_value:
                acceptable_ask = l2_bid
                cross_volume += state.order_depths[product].buy_orders[l2_bid]
                cross_volume += state.order_depths[product].buy_orders[l1_bid]
            elif l1_bid > fair_value:
                acceptable_ask = l1_bid
                cross_volume += state.order_depths[product].buy_orders[l1_bid]
            else:
                acceptable_ask = 10000 + (spread / 2) * 0.8

            if l3_ask < fair_value:
                acceptable_bid = l3_ask
                cross_volume += state.order_depths[product].sell_orders[l3_ask]
                cross_volume += state.order_depths[product].sell_orders[l2_ask]
                cross_volume += state.order_depths[product].sell_orders[l1_ask]
            elif l2_ask < fair_value:
                acceptable_bid = l2_ask
                cross_volume += state.order_depths[product].sell_orders[l2_ask]
                cross_volume += state.order_depths[product].sell_orders[l1_ask]
            elif l1_ask < fair_value:
                acceptable_bid = l1_ask
                cross_volume += state.order_depths[product].sell_orders[l1_ask]
            else:
                acceptable_bid = 10000 - (spread / 2) * 0.8

        # TODO Include bollingers band
        if product == "BANANAS":
            if spread > 2:
                pillow = spread / 2
                alpha, skew = 1, 0
                # alpha setting
                if spread >= 6:
                    alpha = 0.7
                elif spread > 3 and spread < 6:
                    alpha = 1.0
                else:
                    alpha = 1.5

                acceptable_ask = fair_value + pillow * alpha + skew
                acceptable_bid = fair_value - pillow * alpha + skew
            elif spread <= 2:

                # crossing the book
                ratio = l1_bid / l1_ask
                if ratio > 1:
                    fair_value += 1
                else:
                    fair_value -= 1

                if l3_bid > fair_value:
                    acceptable_ask = l3_bid
                elif l2_bid > fair_value:
                    acceptable_ask = l2_bid
                elif l1_bid > fair_value:
                    acceptable_ask = l1_bid
                elif ratio > 1:
                    acceptable_ask = fair_value + 3

                if l3_ask < fair_value:
                    acceptable_bid = l3_ask
                elif l2_ask < fair_value:
                    acceptable_bid = l2_ask
                elif l1_ask < fair_value:
                    acceptable_bid = l1_ask
                elif ratio < 1:
                    acceptable_bid = fair_value - 3

        acceptable_bid = math.ceil(acceptable_bid)
        acceptable_ask = math.floor(acceptable_ask)

        # print("acceptable bid {} ask {} product {}".format(
        #    acceptable_bid, acceptable_ask, product))

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
        if len(self._history) is None:
            self._history = temp_dataframe
        self._history = pd.concat([self._history, temp_dataframe])

    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        """
        Only method required. It takes all buy and sell orders for all symbols as an input,
        and outputs a list of orders to be sent
        """
        result = {}
        print("current state own orders ", state.own_trades)
        # print("current state observation ", state.observations)
        self._process_new_data(state)
        # print("current data ", self._history)

        for product in state.order_depths.keys():
            # order_depth: OrderDepth = state.order_depths[product]
            #put in previous trades to inventory tracker
            self._update_inventory(state, product)

            orders: list[Order] = []

            fair_prices, bullish = self._get_ewm_values_and_indicator(
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
        print(self._current_inventory.iloc[-2:, ])
        print(state.position)
        print('------------------------------------')
        return result
