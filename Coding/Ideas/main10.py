from typing import Dict, List
from datamodel import OrderDepth, TradingState, Order
from pandas import DataFrame
import math
import pandas as pd
import numpy as np

class Trader:
    _history = pd.DataFrame(
            columns=[
                'PEARLS',
                'BANANAS',
                'PINA_COLADAS',
                'COCONUTS',
                'BERRIES',
                'DIVING_GEAR'])
    _history_observation = pd.DataFrame(
            columns=['DOLPHIN_SIGHTINGS'])
    _buy_indicator = {'PINA_COLADAS': False, 'COCONUTS': False,
            'BERRIES': False, 'DIVING_GEAR': False}
    _already_bought = {'PINA_COLADAS': False, 'COCONUTS': False,
            'BERRIES': False, 'DIVING_GEAR': False}
    _sell_indicator = {'PINA_COLADAS': False, 'COCONUTS': False,
            'BERRIES': False, 'DIVING_GEAR': False}
    _already_sold = {'PINA_COLADAS': False, 'COCONUTS': False,
            'BERRIES': False, 'DIVING_GEAR': False}

    _observation_indicator = {
            'DOLPHIN_SIGHTINGS': {"BUY": False, "SELL": False},
            }
    _already_observed = {
            'DOLPHIN_SIGHTINGS': {"BUY": False, "SELL": False},
            }

    def _get_ewm_values_and_indicator(self, state: TradingState, product: str) -> bool:
        """Computes EWM5, EWM12, EWM26, MACD and signaling."""
        history_product = self._history[product]
        current_mid_price = history_product[state.timestamp]

        bullish = -1
        ewm_26, signal, macd = -1, -1, -1
        ewm_5, std_5 = -1, -1
        ewm_8 = -1
        sma_90 = -1

        if state.timestamp > 5 * 100:
            ewm_5 = history_product.ewm(span=5, adjust=False).mean()[state.timestamp]
            std_5 = history_product.ewm(span=5, adjust=False).std()[state.timestamp]

        if state.timestamp > 8 * 100:
            ewm_8 = history_product.ewm(span=8, adjust=False).mean()[state.timestamp]

        if state.timestamp > 26 * 100:
            span_12 = history_product.ewm(span=12, adjust=False).mean()
            span_26 = history_product.ewm(span=26, adjust=False).mean()
            macd_series = span_12 - span_26
            span_9_macd = macd_series.ewm(span=9, adjust=False).mean()

            macd = macd_series[state.timestamp]
            signal = span_9_macd[state.timestamp]

        if state.timestamp > 90 * 100:
            sma_90 = history_product.ewm(span=90).mean()[state.timestamp]


        products_to_choose = [
                'BERRIES',
                'PINA_COLADAS',
                'COCONUTS',
                'DIVING_GEAR']
        products_observation = ['DOLPHIN_SIGHTINGS']

        if product in products_to_choose:
            values = [ewm_5, macd, signal, sma_90]
        else:
            values = [ewm_8, macd, signal, sma_90]

        if product in products_observation:
            already_bought = self._already_observed[product]['BUY']
            already_sold = self._already_observed[product]['SELL']
            if signal < macd: # buy signal
                self._observation_indicator[product]["SELL"] = False
                if not already_bought:
                    self._observation_indicator[product]["BUY"] = True
                    self._already_observed[product]["BUY"] = True
                    self._already_observed[product]["SELL"] = False
                else:
                    self._observation_indicator[product]["BUY"] = False

            elif signal > macd: # sell signal
                self._observation_indicator[product]["BUY"] = False
                if not already_sold:
                    self._observation_indicator[product]["SELL"] = True
                    self._already_observed[product]["SELL"] = True
                    self._already_observed[product]["BUY"] = False
                else:
                    self._observation_indicator[product]["SELL"] = False

        correlate_to_buy, correlate_to_sell = False, False
        if product == 'DIVING_GEAR':
            correlate_to_buy = self._observation_indicator['DOLPHIN_SIGHTINGS']["BUY"]
            correlate_to_sell = self._observation_indicator['DOLPHIN_SIGHTINGS']["SELL"]

        if product in products_to_choose:
            bought_product = self._already_bought[product]
            sold_product = self._already_sold[product]

            if signal < macd: # buy signal
                self._sell_indicator[product] = False
                if not bought_product or correlate_to_buy:
                    self._buy_indicator[product] = True
                    self._already_bought[product] = True
                    self._already_sold[product] = False
                else:
                    self._buy_indicator[product] = False

            elif signal > macd: # sell signal
                self._buy_indicator[product] = False
                if not sold_product or correlate_to_sell:
                    self._sell_indicator[product] = True
                    self._already_sold[product] = True
                    self._already_bought[product] = False
                else:
                    self._sell_indicator[product] = False

            # print("sell indicator for {} {}".format(
            #     product, self._sell_indicator[product]))

        if product in ("BANANAS", "BERRIES") or product in products_to_choose:
            if signal < macd: # bullish
                bullish = True
            elif signal > macd:
                bullish = False

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
        limits = {
                'PEARLS': 20,
                'BANANAS': 20,
                'PINA_COLADAS': 200, # 600
                'COCONUTS': 150, # 300
                'BERRIES': 50, # 250
                'DIVING_GEAR': 20, # 50
                }
        max_long_position = limits[product] - current_volume
        max_short_position = -limits[product] - current_volume

        buy_volume = min(limits[product], max_long_position)
        sell_volume = max(-limits[product], max_short_position)

        # print("acceptable buy vol {} sell vol {} product {}".format(
        #     buy_volume, sell_volume, product))

        return buy_volume, sell_volume

    def cross_price(self, bids, asks, fairValue : float) -> tuple:
        sorted(bids)
        sorted(asks)
        # print("Fair Value = ", fairValue)

        bid_prices = [ask for ask in asks if ask < fairValue]
        bid_price = max(bid_prices) if len(bid_prices) > 0 else -1

        ask_prices = [bid for bid in bids if bid > fairValue]
        ask_price = min(ask_prices) if len(ask_prices) > 0 else -1

        return bid_price, ask_price



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
        acceptable_ask = 1000000000
        asks = sorted(state.order_depths[product].sell_orders)
        bids = sorted(state.order_depths[product].buy_orders)

        l1_bid = bids[-1]
        l1_ask = asks[0]

        spread = l1_ask - l1_bid

        fair_value = fair_prices[0]
        if product == "PEARLS":
            # get sma_90
            fair_value = fair_prices[-1] if fair_prices[-1] > -1 else 10000
            # get sma_90
            temp_bid, temp_ask = self.cross_price(bids=bids, asks=asks, fairValue=fair_value)
            acceptable_bid = 9998 if temp_bid == -1 else temp_bid
            acceptable_ask = 10002 if temp_ask == -1 else temp_ask
            # if acceptable_ask == -1:
            #     acceptable_ask = 10000 + (spread / 2)*0.8
            # if acceptable_bid == -1:
            #     acceptable_bid = 10000 - (spread / 2)*0.8

        if product in [
                "COCONUTS",
                "PINA_COLADAS"]:
            if isinstance(bullish, bool) and bullish:
                # not crossing the books
                # I'm not relying on high frequency
                # but in the good deals provided by the MACD signaling
                # the acceptable prices are adjusted to make sure
                # the bots choose our orders
                # TODO optimize
                acceptable_ask = fair_value + (spread / 4)
                acceptable_bid = fair_value
            elif isinstance(bullish, bool) and not bullish:
                acceptable_ask = fair_value
                acceptable_bid = fair_value - (spread / 4)
            else:
                acceptable_ask = fair_value
                acceptable_bid = fair_value

        #TODO Include bollingers band
        if product in ("BANANAS"):
            if spread > 2:
                pillow = spread / 2
                alpha, skew = 0.8, 0
                acceptable_ask = fair_value + pillow * alpha + skew
                acceptable_bid = fair_value - pillow * alpha + skew
            elif spread <= 2:
                ratio = l1_bid / l1_ask
                fair_value += 1 if ratio > 1 else -1
                pillow = spread / 2
                alpha, skew = 0.8, 0

                # crossing the book
                # Cross price
                temp_bid, temp_ask = self.cross_price(bids=bids, asks=asks, fairValue=fair_value)
                acceptable_bid = 0 if temp_bid == -1 else temp_bid
                acceptable_ask = 10000000 if temp_ask == -1 else temp_ask

                if acceptable_ask == -1 and ratio > 1:
                    acceptable_ask = fair_value + pillow * alpha

                if acceptable_bid == -1 and ratio < 1:
                    acceptable_bid = fair_value - pillow * alpha

        if product in ["DIVING_GEAR","BERRIES"]:
            df = self._history[product]
            timestamp = state.timestamp

            if product == "BERRIES":
              fair_value = df.rolling(5).mean().loc[timestamp]
              temp_bid, temp_ask = self.cross_price(bids=bids, asks=asks, fairValue=fair_value)
              acceptable_bid = 0 if temp_bid == -1 else temp_bid
              acceptable_ask = 100000000 if temp_ask == -1 else temp_ask
            elif product == "DIVING_GEAR":
                acceptable_bid = 0
                acceptable_ask = 100000000

        acceptable_bid = math.ceil(acceptable_bid)
        acceptable_ask = math.floor(acceptable_ask)

        # print("acceptable bid {} ask {} product {}".format(
        #     acceptable_bid, acceptable_ask, product))

        return acceptable_bid, acceptable_ask

    def _process_new_data(self, state: TradingState) -> None:
        """Adds new data point to historical data."""
        # initialize values at the beginning
        if bool(state.observations):
            for product in state.observations.keys():
                value = int(state.observations[product])
            temp_dataframe = pd.DataFrame(
                    [value],
                    columns=[product],
                    index=[state.timestamp])
            if len(self._history_observation) is None:
                self._history_observation = temp_dataframe

            self._history_observation = pd.concat(
                    [self._history_observation, temp_dataframe])

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
            if product in ["DIVING_GEAR","BERRIES"]:
                if (acceptable_ask != 100000000 or acceptable_bid != 0):
                  print(f"{state.timestamp} | bid={acceptable_bid}, ask = {acceptable_ask}")
                # if product == "DIVING_GEAR":
                #     buy_quantity = min(5, buy_quantity)
                #     sell_quantity = max(-5, sell_quantity)
                orders.append(Order(product, acceptable_bid, buy_quantity))
                orders.append(Order(product, acceptable_ask, sell_quantity))
            # if pina_colada or coconuts, place orders if neccesary
            if product in self._buy_indicator.keys():
                buy_product = self._buy_indicator[product]
                sell_product = self._sell_indicator[product]
                if buy_product:
                    orders.append(Order(product, acceptable_bid, buy_quantity))

                if sell_product:
                    orders.append(Order(product, acceptable_ask, sell_quantity))
            else:
                if product in ["DIVING_GEAR","BERRIES"]:
                    print(f"{state.timestamp} | bid={acceptable_bid}, ask = {acceptable_ask}")
                orders.append(Order(product, acceptable_bid, buy_quantity))
                orders.append(Order(product, acceptable_ask, sell_quantity))

            result[product] = orders

        return result
