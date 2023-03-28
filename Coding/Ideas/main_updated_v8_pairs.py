from typing import Dict, List, Any 
from datamodel import OrderDepth, TradingState, Order, ProsperityEncoder, Symbol
from pandas import DataFrame
import math
import pandas as pd
import numpy as np
import json


class Trader:
    _history = pd.DataFrame(columns=['PEARLS', 'BANANAS', 'PINA_COLADAS', 'COCONUTS']) # gets replaced in first iteration

    product_limits = {'PEARLS': 20, 'BANANAS': 20, 'PINA_COLADAS': 300, 'COCONUTS': 600}

    def get_l1_quantity(self, state: TradingState, product: str):
        l1_bid, l1_ask = self.get_l1(state, product)
        l1_bid_quantity = state.order_depths[product].buy_orders[l1_bid]
        l1_ask_quantity = state.order_depths[product].sell_orders[l1_ask]
        return l1_bid_quantity, l1_ask_quantity


    def get_max_quantity(self, state: TradingState, product: str): 
        current_volume = 0
        if bool(state.position): 
            if product in state.position.keys():
                current_volume = state.position[product]

        max_long_position = self.product_limits[product] - current_volume
        max_short_position = -self.product_limits[product] - current_volume

        return max_long_position, max_short_position
    def get_l1(self, state: TradingState, product: str): 
        prod_ask = min(state.order_depths[product].sell_orders)
        prod_bid = max(state.order_depths[product].buy_orders)
        return prod_bid, prod_ask
    def get_mid(self, state: TradingState, product: str): 
        prod_ask = min(state.order_depths[product].sell_orders)
        prod_bid = max(state.order_depths[product].buy_orders)
        return (prod_ask + prod_bid)/2
    def get_sma(self, array: np.array, sma: int) -> float:
        sma_ = np.average(array[-sma:])
        return sma
    
## --------------------------- Start Pairs ---------------------- ## 
    # Pairs Globals
    # def Long Pair long pina, short coco
    # def Short Pair short pina, long coco
    pc_ratio_mean = 1.8732241724483873
    pc_ratio_std = 0.0030937512917090376
    ratio_norm_g = np.array([])
    ratio_g = np.array([])
    cur_ratio = 1.875
    zscore_300_100_g = 0
    trade_active = 'Neutral' # continuosly updated as 'Long Pair' or 'Short Pair' or 'Neutral' to determine if currently long or short a pair
    # Trade pairs strategy 
    def _trade_pairs(self, state: TradingState) ->  bool:
        """
        Return True if a positive pairs signal is detected,
        False if a negative pairs signal is detected,
        -1 if no signal is detected
        """
        signal = -1 # -1 for no signal
        pina_mid = self.get_mid(state, 'PINA_COLADAS')
        coco_mid = self.get_mid(state, 'COCONUTS')

        # get the price ratio
        ratio = pina_mid/coco_mid
        ratio_norm = (ratio - self.pc_ratio_mean)/self.pc_ratio_std
        self.ratio_g = np.append(self.ratio_g, ratio)
        self.ratio_norm_g = np.append(self.ratio_norm_g,ratio_norm) # append to list of current ratio_norms

        # get the ma and signal
        # if len(self.ratio_g) > 300:
        #     ma300 = np.average(self.ratio_g[-400:])
        #     ma100 = np.average(self.ratio_g[-100:])
        #     std_300 = np.std(self.ratio_g[-400:])
        #     zscore_300_100 = (ma100 - ma300)/std_300 # our signal
    
        if ratio > self._pc_ratio_mean + 1.8* self._pc_ratio_std: # short the pair: short pina, long coco
            signal = False # false for short 
        elif ratio < self._pc_ratio_mean - 1.8* self._pc_ratio_std:
            signal = True # true for long
        # self.zscore_300_100_g  = zscore_300_100 # udate for exits
        return signal
    
    def _manage_pairs(self) -> bool:
        closeTrade = False
        if (self.trade_active == 'Long Pair') and (self.ratio_g[-1] >= self._pc_ratio_mean - 1.8* self._pc_ratio_std): # we are currently long a pair
            # must close trade
            closeTrade = True
        elif self.trade_active == 'Short Pair' and (self.ratio_g[-1] <= self._pc_ratio_mean + 1.8 * self._pc_ratio_std):
            # must close trade
            closeTrade = True
        return closeTrade

## --------------------------- End Pairs ---------------------- ## 

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

        values = [history_product.ewm(span=8, adjust=False).mean()[state.timestamp], macd, signal, sma_90]

        if product == "BANANAS" or product == 'COCONUTS' or product == 'PINA_COLADAS':
            if signal < macd: # bullish
                bullish = True
            elif signal > macd:
                bullish = False

        if product == "PEARLS":
            pass

        print("MACD indicator is bullish {} for {}".format(bullish, product))
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
        limits = {'PEARLS': 20, 'BANANAS': 20, 'PINA_COLADAS': 300, 'COCONUTS': 300}
        max_long_position = limits[product] - current_volume
        max_short_position = -limits[product] - current_volume

        buy_volume = min(limits[product], max_long_position)
        sell_volume = max(-limits[product], max_short_position)

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
            #fair_value = 10000
            # still not crossing the books
            #if spread > 3:
            #    if isinstance(bullish, bool) and bullish:
            #        acceptable_bid = l1_bid + 1
            #        acceptable_ask = l1_ask
            #    elif isinstance(bullish, bool) and not bullish:
            #        acceptable_bid = l1_bid
            #        acceptable_ask = l1_ask - 1
            #    else:
            #        acceptable_bid = l1_bid
            #        acceptable_ask = l1_ask
            #elif spread <= 3:
            # crossing the book

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

        #TODO Include bollingers band
        if product == "BANANAS": # product == 'PINA_COLADAS' or product == 'COCONUTS':

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
                ratio = state.order_depths[product].buy_orders[l1_bid] / state.order_depths[product].sell_orders[l1_ask]
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
        print('add ratio market making to higher spreads')
        # print("current state own orders ", state.own_trades)
        # print("current state observation ", state.observations)
        self._process_new_data(state)
        # print("current data ", self._history)

         # boolean if we trade, -1 if we dont.

        for product in state.order_depths.keys():

            # order_depth: OrderDepth = state.order_depths[product]
            orders: list[Order] = []

            fair_prices, bullish = self._get_ewm_values_and_indicator(
                    state, product)
            # pairs trading
            if product == 'PINA_COLADAS': # or product == 'COCONUTS':  only have to do once for pina colads and coconuts
                trade_pairs = self._trade_pairs(state)
                pina_mid = self.get_mid(state, 'PINA_COLADAS')
                coco_mid = self.get_mid(state, 'COCONUTS')
                # get the price ratio
                self.ratio_g = pina_mid / coco_mid
                # set quantities and prices that we may trade
                pina_l1bid_quant, pina_l1ask_quant = self.get_l1_quantity(state, 'PINA_COLADAS')
                coco_l1bid_quant, coco_l1ask_quant = self.get_l1_quantity(state, 'COCONUTS')

                pina_bid, pina_ask = self.get_l1(state, 'PINA_COLADAS')
                coco_bid, coco_ask = self.get_l1(state, 'COCONUTS')

                long_pair_quant = min(-pina_l1ask_quant, math.floor(coco_l1bid_quant/2))
                # Make sure long_pair_quant will not violate position limits
                if (state.position['COCONUTS'] + 2*long_pair_quant >= 600) or \
                    (state.position['PINA_COLADAS'] + long_pair_quant >= 300):
                    long_pair_quant = min(300-state.position['PINA_COLADAS'], math.floor((600-state.position['COCONUTS'])/2), long_pair_quant)
                short_pair_quant = min(pina_l1bid_quant, -math.floor(coco_l1ask_quant/2))
                # Make sure short_pair_quant will not violate position limits
                if (state.position['COCONUTS'] - 2*short_pair_quant <= -600) or \
                    (state.position['PINA_COLADAS'] - short_pair_quant <= -300):
                    short_pair_quant = min(abs(-300-state.position['PINA_COLADAS']), abs(math.ceil((-600-state.position['COCONUTS'])/2)), short_pair_quant)

                # check if we are long pair already, if we are not and have signal to long pair, we long!
                if trade_pairs == True: #and (self.trade_active == 'Neutral' or self.trade_active == 'Short Pair'): l1 from active trade might not be able to fill trade
                    orders.append(Order('PINA_COLADAS', pina_ask, long_pair_quant))
                    orders.append(Order('COCONUTS', coco_bid, -2*long_pair_quant))
                    if (state.position['PINA_COLADAS'] +long_pair_quant > 0) and (state.position['COCONUTS'] - short_pair_quant*2 < 0):
                        self.trade_active = 'Long Pair' # update if we are in trade or not
                # check if we must close trade
                elif trade_pairs == -1 and self.trade_active == 'Long Pair':
                    manage_pairs = self._manage_pairs() 
                    if manage_pairs: # we have to close trade
                        if bool(state.position):
                            if product in state.position.keys():
                                short_pair_quant = min(short_pair_quant, state.position['PINA_COLADAS'], -math.ceil(state.position['COCONUTS']/2))
                                orders.append(Order('PINA_COLADAS', pina_bid, -short_pair_quant)) # SELL OUR PINAS at bid
                                orders.append(Order('COCONUTS', coco_bid, short_pair_quant*2)) # BUY OUR COCOS at ask
                                if (state.position['PINA_COLADAS'] - short_pair_quant == 0) and (state.position['COCONUTS'] + short_pair_quant*2 == 0):
                                    self.trade_active = 'Neutral'
                        
                # check if we are short a pair already, if we are not short a pair and have a signal, we short!       
                elif trade_pairs == False: #and (self.trade_active == 'Neutral' or self.trade_active == 'Long Pair'):
                    orders.append(Order('PINA_COLADAS', pina_bid, -short_pair_quant))
                    orders.append(Order('COCONUTS', coco_ask, 2*short_pair_quant))
                    if (state.position['PINA_COLADAS'] - short_pair_quant < 0) and (state.position['COCONUTS'] + short_pair_quant*2 > 0):
                        self.trade_active = 'Short Pair' # update if we are in trade or not
                # check if we must close trade
                elif trade_pairs == -1 and self.trade_active == 'Short Pair':
                    manage_pairs = self._manage_pairs()
                    if manage_pairs: # we have to close trade if true
                        if bool(state.position):
                            if product in state.position.keys():
                                long_pair_quant = min(long_pair_quant,
                                                      -state.position['PINA_COLADAS'],
                                                      math.floor(state.position['COCONUTS'] / 2))
                                orders.append(Order('PINA_COLADAS', pina_ask, long_pair_quant)) # BUY OUR PINAS at ask
                                orders.append(Order('COCONUTS', coco_bid, -long_pair_quant*2)) # SELL OUR COCOS at bid
                                if (state.position['PINA_COLADAS'] - short_pair_quant == 0) and (state.position['COCONUTS'] + short_pair_quant*2 == 0):
                                    self.trade_active = 'Neutral'
            # else:
            #     acceptable_bid, acceptable_ask = self._get_acceptable_price(
            #         state, product, fair_prices, bullish)
            #     # get quantities to place orders
            #     buy_quantity, sell_quantity = self._get_acceptable_quantity(
            #             state, product, bullish)
            #     # place orders
            #     orders.append(Order(product, acceptable_bid, buy_quantity))
            #     orders.append(Order(product, acceptable_ask, sell_quantity))

            result[product] = orders

        return result
