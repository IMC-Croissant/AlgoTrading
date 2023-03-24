from typing import Dict, List
from datamodel import OrderDepth, TradingState, Order
from pandas import DataFrame
import math
import pandas as pd
import numpy as np

class Trader:
    _history = pd.DataFrame(columns=['PEARLS', 'BANANAS', 'PINA_COLADAS', 'COCONUTS']) # gets replaced in first iteration
    _buy_indicator = {'PINA_COLADAS': False, 'COCONUTS': False}
    _sell_indicator = {'PINA_COLADAS': False, 'COCONUTS': False}

    def get_l1(self, state: TradingState, product: str): 
        prod_ask = min(state.order_depths[product].sell_orders)
        prod_bid = max(state.order_depths[product].buy_orders)
        return prod_bid, prod_ask
    def _get_ewm_values_and_indicator(self, state: TradingState, product: str) -> bool:
        """Computes EWM5, EWM12, EWM26, MACD and signaling."""
        def sma_cross_indicator(self, state: TradingState, product: str, history_product, sma_low, sma_high) -> None:
            """
            Calculate SMA
            """
            if state.timestamp > sma_high * 100:
                sma_low = history_product.iloc[-sma_low:].mean()
                sma_high = history_product.iloc[-sma_high:].mean()
            if sma_low != -1 and sma_high != -1:
                if sma_low > sma_high:
                    self._sell_indicator[product] = False
                    self._buy_indicator[product] = True
                elif sma_low < sma_high:
                    self._buy_indicator[product] = False
                    self._sell_indicator[product] = True
        def
        history_product = self._history[product]
        current_mid_price = history_product[state.timestamp]

        bullish = -1
        ewm_5, ewm_26, signal, macd = -1, -1, -1, -1
        ewm_8 = -1
        sma_20, sma_50, sma_90 = -1, -1, -1
        sma_low, sma_high = -1, -1
        ewm_5 = history_product.ewm(span=5, adjust=False).mean()[state.timestamp]
        ewm_8 = history_product.ewm(span=8, adjust=False).mean()[state.timestamp]


        # For ewm and MACD calculations
        if state.timestamp > 26 * 100:
            span_12 = history_product.ewm(span=12, adjust=False).mean()
            span_26 = history_product.ewm(span=26, adjust=False).mean()
            macd_series = span_12 - span_26
            span_9_macd = macd_series.ewm(span=9, adjust=False).mean()

            macd = macd_series[state.timestamp]
            signal = span_9_macd[state.timestamp]

        if state.timestamp > 90 * 100:
            sma_90 = history_product.ewm(span=90).mean()[state.timestamp]


        # USING MACD
        #if product in ('PINA_COLADAS', 'COCONUTS'):
        #    if signal < macd: # buy signal
        #        self._sell_indicator[product] = False
        #        self._buy_indicator[product] = True
        #    elif signal > macd: # sell signal
        #        self._buy_indicator[product] = False
        #        self._sell_indicator[product] = True
        #    print("sell indicator for {} {}".format(
        #        product, self._sell_indicator[product]))
        # USING SMA cross
        if product == 'PINA_COLADAS':
            # For SMA calculations
            sma_cross_indicator(self, state, product, history_product, 15, 40)
        if product  == 'COCONUTS':
            sma_cross_indicator(self, state, product, history_product, 20, 50)
        if product == "BANANAS":
            pass

        if product == 'PINA_COLADAS':
            values = [ewm_5, macd, signal, sma_90]
        elif product == 'COCONUTS':
            values = [ewm_5, macd, signal, sma_90]
        else:
            values = [ewm_8, macd, signal, sma_90]
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
        limits = {'PEARLS': 20, 'BANANAS': 20, 'PINA_COLADAS': 300, 'COCONUTS': 600}
        max_long_position = limits[product] - current_volume
        max_short_position = -limits[product] - current_volume

        if
        buy_volume = max_long_position
        sell_volume = max_short_position
        #buy_volume = min(limits[product], max_long_position)
        #sell_volume = max(-limits[product], max_short_position)


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

        fair_value = fair_prices[0]

        if product == "PEARLS":
            # get sma_90
            fair_value = fair_prices[-1] if fair_prices[-1] > -1 else 10000
            if l3_bid > fair_value:
                acceptable_ask = l3_bid
            elif l2_bid > fair_value:
                acceptable_ask = l2_bid
            elif l1_bid > fair_value:
                acceptable_ask = l1_bid
            else:
                acceptable_ask = 10000 + (spread / 2)*0.8

            if l3_ask < fair_value:
                acceptable_bid = l3_ask
            elif l2_ask < fair_value:
                acceptable_bid = l2_ask
            elif l1_ask < fair_value:
                acceptable_bid = l1_ask
            else:
                acceptable_bid = 10000 - (spread / 2)*0.8

        elif product == "COCONUTS" or product == "PINA_COLADAS":
            if self._buy_indicator[product]:
                acceptable_bid = l1_ask + 1
            elif self._sell_indicator[product]:
                acceptable_ask = l1_bid - 1
            #else:
            #    acceptable_ask = fair_value + (spread /2 *.8)
            #    acceptable_bid = fair_value - (spread /2 *.8)


        elif product == "BANANAS":

            if spread > 2:
                pillow = spread / 2
                alpha, skew = 0.8, 0
                #alpha setting
                #if spread >= 6:
                #    alpha = 0.8
                #elif spread > 3 and spread < 6:
                #    alpha = 1
                #else:
                #    alpha = 1.5

                acceptable_ask = fair_value + pillow * alpha + skew
                acceptable_bid = fair_value - pillow * alpha + skew
            elif spread <= 2:
                ratio = l1_bid / l1_ask
                if ratio > 1:
                    fair_value += 1
                else:
                    fair_value -= 1
                pillow = spread / 2
                alpha, skew = 0.8, 0

                # crossing the book
                if l3_bid > fair_value:
                    acceptable_ask = l3_bid
                elif l2_bid > fair_value:
                    acceptable_ask = l2_bid
                elif l1_bid > fair_value:
                    acceptable_ask = l1_bid
                elif ratio > 1:
                    acceptable_ask = fair_value + pillow * alpha

                if l3_ask < fair_value:
                    acceptable_bid = l3_ask
                elif l2_ask < fair_value:
                    acceptable_bid = l2_ask
                elif l1_ask < fair_value:
                    acceptable_bid = l1_ask
                elif ratio < 1:
                    acceptable_bid = fair_value - pillow * alpha


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
        if len(self._history) is None:
            self._history = temp_dataframe
        self._history = pd.concat([self._history, temp_dataframe])


    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        """
        Only method required. It takes all buy and sell orders for all symbols as an input,
        and outputs a list of orders to be sent
        """
        result = {}
        # print('add ratio market making to higher spreads')
        # print("current state own orders ", state.own_trades)
        # print("current state observation ", state.observations)
        self._process_new_data(state)
        # print("current data ", self._history)

        for product in state.order_depths.keys():
            # order_depth: OrderDepth = state.order_depths[product]
            orders: list[Order] = []

            fair_prices, bullish = self._get_ewm_values_and_indicator(
                    state, product)

            acceptable_bid, acceptable_ask = self._get_acceptable_price(
                state, product, fair_prices, bullish)
            # get quantities to place orders
            buy_quantity, sell_quantity = self._get_acceptable_quantity(
                    state, product, bullish)
            # if pina_colada or coconuts, place orders if neccesary

            if product in self._buy_indicator.keys():
                buy_product = self._buy_indicator[product]
                sell_product = self._sell_indicator[product]
                if buy_product:
                    orders.append(Order(product, acceptable_bid, buy_quantity))

                if sell_product:
                    orders.append(Order(product, acceptable_ask, sell_quantity))
            else:
                orders.append(Order(product, acceptable_bid, buy_quantity))
                orders.append(Order(product, acceptable_ask, sell_quantity))

            result[product] = orders

        return result
