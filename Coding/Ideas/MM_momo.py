from typing import Dict, List
from datamodel import OrderDepth, TradingState, Order
import pandas as pd
from pandas import DataFrame
import math

class Trader:


## Make a market algo ## 
    def make_a_market(self, asks: list, bids: list, momoFlag: int, product, fairvalue) -> tuple:
        
        mm_bid = 0 
        mm_ask = 10000000

        l1_ask = asks[0]
        l1_bid = bids[-1]

        l2_bid = 0
        l3_bid = 0

        l2_ask = 1000000
        l3_ask = 1000000

        # For crossing the book we need to check if there are multiple levels we can win on
        # Thus we look at all lvls of the book
        if len(bids) > 2:
            l3_bid = bids[-3]
            l2_bid = bids[-2]
        elif len(bids) > 1:
            l2_bid = bids[-2]
        
        if len(asks) > 2:
            l3_ask = asks[2]
            l2_ask = asks[1]
        elif len(asks) > 1:
            l2_ask = asks[1]
    

        spread = l1_ask-l1_bid
        if product == 'PEARLS':
            thre = 3
            momoFlag = -1
        
        if product == 'BANANAS':
            thre = 2
            momoFlag = -1

        if spread > thre:        
            if momoFlag == 1: # Bullish Trend --> aggresive bids
                mm_bid = l1_bid + 1
                mm_ask = l1_ask 
            elif momoFlag == 0: # Bearish Trend -> aggressive ask
                mm_bid = l1_bid 
                mm_ask = l1_ask - 1
            elif momoFlag == -1: # No trend -> L1 bid and ask
                mm_bid = l1_bid 
                mm_ask = l1_ask 
        elif product == 'PEARLS' and spread <= thre: # liquid market with FV cross
            if l3_bid > 10000:
                mm_ask = l3_bid # cross the book (sell above FV)
            elif l2_bid > 10000:
                mm_ask = l2_bid
            elif l1_bid > 10000:
                mm_ask = l1_bid
            elif l3_ask < 10000:
                mm_bid = l3_ask # cross the book (buy below FV)
            elif l2_ask < 10000:
                mm_bid = l2_ask
            elif l1_ask < 10000:
                mm_bid = l1_ask
        elif product == 'BANANAS' and spread <= thre: # Cross the book but lets look at 
            if l3_bid > fairvalue:
                mm_ask = l3_bid
            elif l2_bid > fairvalue:
                mm_ask = l2_bid
            elif l1_bid > fairvalue:
                mm_ask = l1_bid
            elif l3_ask < fairvalue:
                mm_bid = l3_ask
            elif l2_ask < fairvalue:
                mm_bid = l2_ask
            elif l1_ask < fairvalue:
                mm_bid = l1_ask
        mm_bid = math.ceil(mm_bid)
        mm_ask = math.floor(mm_ask)
        return mm_bid, mm_ask

## -- QUANTITY CONTROL -- ## 
    def get_quantity(self, product: str, max_long: int, max_short: int, cur_pos: int, momo_flag: int) -> tuple:

        buy_quantity = min(max_long, 20) # max_long can be > 20 we dont ever want to 
        sell_quantity = max(max_short, -20)

        # For trendy product we adjust quantity 
        if product == 'BANANAS':
            if momo_flag == 1: #bull trend, so we want to buy more than we sell
                sell_quantity = sell_quantity - 1
                buy_quantity = buy_quantity 
            elif momo_flag == 0: #bear trend, so we want to sell more than we buy 
                sell_quantity = sell_quantity
                buy_quantity = buy_quantity - 1
                        
        # We safeguard against trend going against us accordingly 
            if cur_pos > 12 and momo_flag == 0:  # bear trend
                sell_quantity += 1
                buy_quantity
            if cur_pos < -12 and momo_flag == 1: # bull trend
                sell_quantity
                buy_quantity += 1
                
            return buy_quantity, sell_quantity
        elif product == 'PEARLS':
            return buy_quantity, sell_quantity
        
        
    
## -- HISTORY DATAFRAME -- ##
    _history = pd.DataFrame([[10000, 4948]], columns= ['PEARLS', 'BANANAS'], index = [0])

## -- INVENTORY MANAGER -- ##
    # def inventory_manager(self, product: str):
    #     phi()

    def _get_cumavg(self, state: TradingState, product: str) -> float:
        history_product = self._history[product]
        if state.timestamp > 100:
            cum_avg = history_product.rolling(window = len(history_product)).mean()[state.timestamp] # we record cumulative avg of price
        else: 
            cum_avg = -1
        return cum_avg
## SMA ##  
    def _get_sma(self, state: TradingState, product: str) -> tuple:
        """Computes SMA20 and SMA50 from historical data"""
        history_product = self._history[product]
        
        # if state.timestamp > 5100:
        if state.timestamp > 5000:
            sma_20 = history_product.rolling(window = 5).mean()[state.timestamp]
            sma_50 = history_product.rolling(window = 50).mean()[state.timestamp]
            sma_5 = history_product.rolling(window = 5).mean()[state.timestamp]
            return sma_5, sma_20, sma_50
        elif state.timestamp > 500:
            sma_5 = history_product.rolling(window = 5).mean()[state.timestamp]
            return sma_5, -1, -1 
        else:
            return -1, -1, -1

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

## HISTORICAL FOR SMA COMPUTATION ##
    def _process_new_data(self, state: TradingState) -> None:
        """Adds new data point to historical data."""
        # initialize values at the beginning
        # own_trades still not populated
        # if ("PEARLS" not in state.market_trades.keys() or "BANANAS" not in state.market_trades.keys()):
        #     pass
        # else:
        # Get the midprices and append to our dataframe
        mid_prices = []
        for key in state.order_depths.keys():
            l1_ask = min(state.order_depths[key].sell_orders)
            l1_bid = max(state.order_depths[key].buy_orders)
            cur_mid = (l1_ask + l1_bid)/2
            mid_prices.append(cur_mid)

        market_trades = state.market_trades

        our_columns = [key for key in state.order_depths.keys()]
        temp_df = pd.DataFrame([mid_prices], columns = our_columns, index = [state.timestamp])
        self._history = pd.concat([self._history, temp_df])

## Run ##
    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        """
        Only method required. It takes all buy and sell orders for all symbols as an input,
        and outputs a list of orders to be sent
        """
        # Initialize the method output dict as an empty dict for result
        result = {}

        # Initialize market trades, own trades, and current position
        market_trades = state.market_trades
        own_trades = state.own_trades
        position = state.position

        # Store data from current timestamp
        self._process_new_data(state)

        # Iterate over all the keys (the available products) contained in the order depths
        for product in state.order_depths.keys():
            # Retrieve the Order Depth containing all the market BUY and SELL orders for PEARLS
            order_depth: OrderDepth = state.order_depths[product]

            # Initialize the list of Orders to be sent as an empty list
            orders: list[Order] = []
        
            # Safeguard for every order to execute
            cur_pos = 0
            
            if bool(position):
                if product in position.keys():
                    cur_pos = state.position[product]
            max_long = 20 - cur_pos # cannot buy more than this
            max_short = -20 - cur_pos # cannot short more than this

            # Momentum Flag: if 20SMA >< 50SMA inventory & spreads change accordingly. Use HURST exp to ID if trendy
            sma_5, sma_20, sma_50 = self._get_sma(state, product)
            cum_avg = self._get_cumavg(state, product)

            if sma_50 == -1:
                momo_flag = -1 # Not enough history just make regular market
            elif sma_50 < sma_20:
                momo_flag = 1 # Bullish Trend
            else:
                momo_flag = 0 # Bearish Trend

            if product == 'PEARLS':
                momo_flag = -1

            # LEVEL 1: ask and bid
            asks = sorted(order_depth.sell_orders.keys())
            bids = sorted(order_depth.buy_orders.keys())
            
            fairvalue = sma_5 # this is always our fair value

            if cum_avg == -1:
                fairvalue = 4928
            elif sma_20 == -1:
                fairvalue = cum_avg # let fair value be the avg of the first couple days
            else: 
                fairvalue == sma_20

            # MAKE THE MARKET
            mm_bid, mm_ask = self.make_a_market(asks, bids, momo_flag, product, fairvalue) # assign our bid/ask spread

            # INVENTORY MANAGEMENT/VOLUME             
            buy_quantity, sell_quantity = self.get_quantity(product, max_long, max_short, cur_pos, momo_flag)
            buy_quantity = min(buy_quantity, 20) # max_long can be > 20 we dont ever want to 
            sell_quantity = max(sell_quantity, -20)
            # ORDER UP!
            # if product == 'PEARLS'/'BANANAS':
            orders.append(Order(product, mm_bid, buy_quantity))
            orders.append(Order(product, mm_ask, sell_quantity))

            result[product] = orders
            
            print("state.own_trades = ", own_trades)
            print("state.market_trades = ", market_trades)
            print("state.position = ", position)
            print("orders placed = ", orders)

        return result
