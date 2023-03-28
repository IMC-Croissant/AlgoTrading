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
                'DIVING_GEAR',
                'BAGUETTE',
                'PICNIC_BASKET',
                'DIP',
                'UKULELE'])
    _history_observation = pd.DataFrame(
            columns=['DOLPHIN_SIGHTINGS'])
    _buy_indicator = {
            'PINA_COLADAS': False,
            'COCONUTS': False,
            'BERRIES': False,
            'DIVING_GEAR': False,
            'BAGUETTE': False,
            'PICNIC_BASKET': False,
            'DIP': False,
            'UKULELE': False}
    _already_bought = {
            'PINA_COLADAS': False,
            'COCONUTS': False,
            'BERRIES': False,
            'DIVING_GEAR': False,
            'BAGUETTE': False,
            'PICNIC_BASKET': False,
            'DIP': False,
            'UKULELE': False}
    _sell_indicator = {
            'PINA_COLADAS': False,
            'COCONUTS': False,
            'BERRIES': False,
            'DIVING_GEAR': False,
            'BAGUETTE': False,
            'PICNIC_BASKET': False,
            'DIP': False,
            'UKULELE': False}
    _already_sold = {
            'PINA_COLADAS': False,
            'COCONUTS': False,
            'BERRIES': False,
            'DIVING_GEAR': False,
            'BAGUETTE': False,
            'PICNIC_BASKET': False,
            'DIP': False,
            'UKULELE': False}

    _observation_indicator = {
            'DOLPHIN_SIGHTINGS': {"BUY": False, "SELL": False},
            }
    _already_observed = {
            'DOLPHIN_SIGHTINGS': {"BUY": False, "SELL": False},
            }

    _crossed_the_book_for_ask = False
    _crossed_the_book_for_bid = False

    def get_l1_quantity(self, state: TradingState, product: str):
        l1_bid, l1_ask = self.get_l1(state, product)
        l1_bid_quantity = state.order_depths[product].buy_orders[l1_bid]
        l1_ask_quantity = state.order_depths[product].sell_orders[l1_ask]
        return l1_bid_quantity, l1_ask_quantity

    def get_l1_l2_quantity(self, state: TradingState, product: str):
        asks = sorted(state.order_depths[product].sell_orders)
        bids = sorted(state.order_depths[product].buy_orders)
        pass

    product_limits = {'PEARLS': 20, 'BANANAS': 20, 'PINA_COLADAS': 300, 'COCONUTS': 600}
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
        return (prod_ask + prod_bid) / 2

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
    trade_active = 'Neutral'  # continuosly updated as 'Long Pair' or 'Short Pair' or 'Neutral' to determine if currently long or short a pair

    # Trade pairs strategy
    def _trade_pairs(self, state: TradingState) -> bool:
        """
        Return True if a positive pairs signal is detected,
        False if a negative pairs signal is detected,
        -1 if no signal is detected
        """
        signal = -1  # -1 for no signal
        pina_mid = self.get_mid(state, 'PINA_COLADAS')
        coco_mid = self.get_mid(state, 'COCONUTS')

        # get the price ratio
        ratio = pina_mid / coco_mid
        ratio_norm = (ratio - self.pc_ratio_mean) / self.pc_ratio_std
        self.ratio_g = ratio
        self.ratio_norm_g = ratio_norm  # append to list of current ratio_norms

        # get the ma and signal
        # if len(self.ratio_g) > 300:
        #     ma300 = np.average(self.ratio_g[-400:])
        #     ma100 = np.average(self.ratio_g[-100:])
        #     std_300 = np.std(self.ratio_g[-400:])
        #     zscore_300_100 = (ma100 - ma300)/std_300 # our signal

        if ratio > self.pc_ratio_mean + 1.5 * self.pc_ratio_std:  # short the pair: short pina, long coco
            signal = False  # false for short
        elif ratio < self.pc_ratio_mean - 1.5 * self.pc_ratio_std:
            signal = True  # true for long
        # self.zscore_300_100_g  = zscore_300_100 # udate for exits
        return signal


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
            'PINA_COLADAS',
            'COCONUTS',
            'BERRIES',
            'DIVING_GEAR',
            'BAGUETTE',
            'PICNIC_BASKET',
            'DIP',
            'UKULELE']

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
                'PINA_COLADAS': 200, # 300
                'COCONUTS': 600, # 600
                'BERRIES': 50, # 250
                'DIVING_GEAR': 20, # 50
                'BAGUETTE': 150, # 150
                'PICNIC_BASKET': 70, # 70
                'DIP': 300, # 300
                'UKULELE': 20, # 70
                }

        max_long_position = limits[product] - current_volume
        max_short_position = -limits[product] - current_volume

        buy_volume = min(limits[product], max_long_position)
        sell_volume = max(-limits[product], max_short_position)


        # print("acceptable buy vol {} sell vol {} product {}".format(
        #     buy_volume, sell_volume, product))

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
            # crossing the book
            if l3_bid > fair_value:
                acceptable_ask = l3_bid
            elif l2_bid > fair_value:
                acceptable_ask = l2_bid
            elif l1_bid > fair_value:
                acceptable_ask = l1_bid
            else:
                acceptable_ask = fair_value + (spread / 2)*0.8

            if l3_ask < fair_value:
                acceptable_bid = l3_ask
            elif l2_ask < fair_value:
                acceptable_bid = l2_ask
            elif l1_ask < fair_value:
                acceptable_bid = l1_ask
            else:
                acceptable_bid = fair_value - (spread / 2)*0.8

        spreads_dict = {
                'PINA_COLADAS': spread / 6,
                'COCONUTS': spread / 6,
                'BERRIES': spread / 6,
                'DIVING_GEAR': spread / 10,
                'BAGUETTE': spread / 2,
                'PICNIC_BASKET': spread / 2,
                'DIP': spread / 6,
                'UKULELE': spread / 2
                }
        if product in spreads_dict.keys():

            # crossing the book
            if l3_bid > fair_value:
                acceptable_ask = l3_bid
                self._crossed_the_book_for_ask = True
            elif l2_bid > fair_value:
                acceptable_ask = l2_bid
                self._crossed_the_book_for_ask = True
            elif l1_bid > fair_value:
                acceptable_ask = l1_bid
                self._crossed_the_book_for_ask = True
            else:
                if isinstance(bullish, bool) and bullish:
                    # TODO optimize
                    acceptable_ask = fair_value + spreads_dict[product]
                elif isinstance(bullish, bool) and not bullish:
                    acceptable_ask = fair_value - spreads_dict[product]
                else:
                    acceptable_ask = fair_value

            if l3_ask < fair_value:
                acceptable_bid = l3_ask
                self._crossed_the_book_for_bid = True
            elif l2_ask < fair_value:
                acceptable_bid = l2_ask
                self._crossed_the_book_for_bid = True
            elif l1_ask < fair_value:
                acceptable_bid = l1_ask
                self._crossed_the_book_for_bid = True
            else:
                if isinstance(bullish, bool) and bullish:
                    # the bots choose our orders
                    # TODO optimize
                    acceptable_bid = fair_value + spreads_dict[product]
                elif isinstance(bullish, bool) and not bullish:
                    acceptable_bid = fair_value - spreads_dict[product]
                else:
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
        state_position_fixed = {}
        for i in ['PINA_COLADAS', 'COCONUTS']:
            if i in state.position:
                state_position_fixed[i] = state.position[i]
            else:
                state_position_fixed[i] = 0

        # boolean if we trade, -1 if we dont.
        result['PINA_COLADAS'] = []
        result['COCONUTS'] = []
        for product in ['PINA_COLADAS']:  # or product == 'COCONUTS':  only have to do once for pina colads and coconut
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

            long_pair_quant = min(-pina_l1ask_quant, math.floor(coco_l1bid_quant / 2))
            # Make sure long_pair_quant will not violate position limits
            if (state_position_fixed['COCONUTS'] - 2 * long_pair_quant <= -600) or \
                    (state_position_fixed['PINA_COLADAS'] + long_pair_quant >= 300):
                long_pair_quant = min(300 - state_position_fixed['PINA_COLADAS'],
                                      abs(math.ceil((-600 - state_position_fixed['COCONUTS']) / 2)),
                                      long_pair_quant)
            short_pair_quant = min(pina_l1bid_quant, -math.floor(coco_l1ask_quant / 2))
            # Make sure short_pair_quant will not violate position limits
            if (state_position_fixed['COCONUTS'] + 2 * short_pair_quant >= 600) or \
                    (state_position_fixed['PINA_COLADAS'] - short_pair_quant <= -300):
                short_pair_quant = min(abs(-300 - state_position_fixed['PINA_COLADAS']),
                                       abs(math.ceil((-600 - state_position_fixed['COCONUTS']) / 2)), short_pair_quant)

            # check if we are long pair already, if we are not and have signal to long pair, we long!
            if trade_pairs == True:  # and (self.trade_active == 'Neutral' or self.trade_active == 'Short Pair'): l1 from active trade might not be able to fill trade
                print(f'going long with long pair quant = {long_pair_quant}')
                result['PINA_COLADAS'].append(Order('PINA_COLADAS', pina_ask + 5, long_pair_quant))
                result['COCONUTS'].append(Order('COCONUTS', coco_bid, -2 * long_pair_quant))
                if (state_position_fixed['PINA_COLADAS'] + long_pair_quant > 0) and (
                        state_position_fixed['COCONUTS'] - short_pair_quant * 2 < 0):
                    self.trade_active = 'Long Pair'  # update if we are in trade or not

            # check if we are short a pair already, if we are not short a pair and have a signal, we short!
            elif trade_pairs == False:  # and (self.trade_active == 'Neutral' or self.trade_active == 'Long Pair'):
                print(f'going short with short pair quant = {short_pair_quant}')
                result['PINA_COLADAS'].append(Order('PINA_COLADAS', pina_bid - 5, -short_pair_quant))
                result['COCONUTS'].append(Order('COCONUTS', coco_ask, 2 * short_pair_quant))
                if (state_position_fixed['PINA_COLADAS'] - short_pair_quant < 0) and (
                        state_position_fixed['COCONUTS'] + short_pair_quant * 2 > 0):
                    self.trade_active = 'Short Pair'  # update if we are in trade or not

        for product in ['PEARLS', 'BANANAS',
                        'BERRIES',#'DIVING_GEAR',
                        'BAGUETTE','PICNIC_BASKET',
                        'DIP','UKULELE']:
            # order_depth: OrderDepth = state.order_depths[product]
            orders: list[Order] = []
            fair_prices, bullish = self._get_ewm_values_and_indicator(
                    state, product)
            acceptable_bid, acceptable_ask = self._get_acceptable_price(
                state, product, fair_prices, bullish)
            # get quantities to place orders
            buy_quantity, sell_quantity = self._get_acceptable_quantity(
                    state, product, bullish)
            # if pina_colada or coconuts, place orders if neccesary using pair trade code vomit
            if product in ['BERRIES',#'DIVING_GEAR',
                        'BAGUETTE','PICNIC_BASKET',
                        'DIP','UKULELE'] and product not in ['PINA_COLADAS', 'COCONUTS']:
                buy_product = self._buy_indicator[product]
                sell_product = self._sell_indicator[product]
                if buy_product or self._crossed_the_book_for_bid:
                    orders.append(Order(product, acceptable_bid, buy_quantity))
                if sell_product: #or not self._crossed_the_book_for_ask:
                    orders.append(Order(product, acceptable_ask, sell_quantity))
            elif product not in ['PINA_COLADAS', 'COCONUTS']:
                orders.append(Order(product, acceptable_bid, buy_quantity))
                orders.append(Order(product, acceptable_ask, sell_quantity))

            result[product] = orders

        return result

