from typing import Dict, List
from datamodel import OrderDepth, TradingState, Order
import math
import numpy as np
import pandas as pd


class Trader:
    count_position_exceded = 0
    # first_fairValue_banana = 4928
    count_iteration = 0
    data_history = {
        "BANANAS": [],
        "PEARLS" : []
    }
    # data_history_ = []
## Predefined values
    position_limit = 20


## next volume
    def next_volume(self, curr_pos : int) ->tuple:
        # n = 0.005 # shape parameter
        if curr_pos < 0:
            bid_vol = self.position_limit
            ask_vol = -1 * self.position_limit - curr_pos
            # ask_vol = ask_vol * np.exp(-1 * abs(curr_pos) * n)
        else:
            ask_vol = -1 * self.position_limit
            bid_vol = self.position_limit - curr_pos
            # bid_vol =  bid_vol * np.exp(-1 * abs(curr_pos) * n)
        return np.ceil(bid_vol), np.floor(ask_vol)
    

## Make a market algo ## 
    def make_a_market(self, state: TradingState , bids, asks, product : str, trend  = -1) -> tuple:
        mm_bid = l1_bid = bids[-1]
        mm_ask = l1_ask = asks[0]
        spread = l1_ask-l1_bid
        # trend = -1
        spread_optimal = {"BANANAS" : 2, "PEARLS" : 3}
        
        # if spread >= 3:
        #     if trend == 1: # Bullish
        #         mm_bid = l1_bid + 1
        #     elif trend == 0: # Berish trend -> sell more 
        #         mm_ask = l1_ask - 1
        if spread <= spread_optimal[product]:
            if product == "BANANAS":
                fairValue = self.get_fair_value(state=state, product=product)
                temp_bid, temp_ask = self.cross_price(bids=bids, asks=asks, product=product, fairValue=fairValue)
                mm_bid = 0 if temp_bid == -1 else temp_bid
                mm_ask = 1000000 if temp_ask == -1 else temp_ask
        
            elif product == "PEARLS":
                fairValue = 10000
                temp_bid, temp_ask = self.cross_price(bids=bids, asks=asks, product=product, fairValue=fairValue)
                mm_bid = 0 if temp_bid == -1 else temp_bid
                mm_ask = 1000000 if temp_ask == -1 else temp_ask

        mm_bid = math.ceil(mm_bid)
        mm_ask = math.floor(mm_ask)
        return mm_bid, mm_ask
    
    def get_fair_value(self, state : TradingState, product : str):
        timestamp = state.timestamp
        n = int(timestamp / 100) + 1
        if n < 15:
            fairValue = np.mean(self.data_history[product][:n])
        else:
            fairValue = np.mean(self.data_history[product][-15:])
        return fairValue
    
    def cross_price(self, bids, asks, product : str, fairValue : float) -> tuple:
        sorted(bids)
        sorted(asks)
        print("Fair Value = ", fairValue)

        bid_prices = [ask for ask in asks if ask < fairValue]
        bid_price = max(bid_prices) if len(bid_prices) > 0 else -1

        ask_prices = [bid for bid in bids if bid > fairValue]
        ask_price = min(ask_prices) if len(ask_prices) > 0 else -1

        return bid_price, ask_price

    def get_trend(self, state : TradingState, product : str) -> int: # 1-> Bullish, 2->Bearish
        timestamp = state.timestamp
        if timestamp > 50 * 100:
            sma50 = np.mean(self.data_history[product][-50:])
            sma20 = np.mean(self.data_history[product][-20:])
            return sma50 < sma20
        else:
            return -1

## Run ##
    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        """
        Only method required. It takes all buy and sell orders for all symbols as an input,
        and outputs a list of orders to be sent
        """
        # Initialize the method output dict as an empty dict for result
        result = {}

        # Store new data to dataframe
        products = state.order_depths.keys()
        # self._process_new_data(state=state, products=products)

        # Initialize market trades, own trades, and current position
        market_trades = state.market_trades
        own_trades = state.own_trades
        position = state.position
        orders: list[Order] = []

        # Iterate over all the keys (the available products) contained in the order depths
        for product in state.order_depths.keys():
            # if product == "PEARLS":

            # Retrieve the Order Depth containing all the market BUY and SELL orders for PEARLS
            order_depth: OrderDepth = state.order_depths[product]

            # Initialize the list of Orders to be sent as an empty list
            orders = []

            bids = sorted(order_depth.buy_orders.keys())
            asks = sorted(order_depth.sell_orders.keys())

            l1_bid = bids[-1]
            l1_ask = asks[0]

            # 
            self.data_history[product].append((l1_ask + l1_bid) / 2)
            # print("DATA = ", self.data_history_bananas)

            # trend = self.get_trend(state=state, product=product)
            mm_bid, mm_ask = self.make_a_market(state=state, bids=bids, asks=asks, product= product)

            curr_pos = 0
            if product in state.position:
                curr_pos = state.position[product]
            
            # bid vol btw 0 to 20 and ask vol btw -20 to 0
            bid_vol, ask_vol = self.next_volume(curr_pos=curr_pos)

            
            orders.append(Order(product, mm_bid, bid_vol))
            orders.append(Order(product, mm_ask, ask_vol))


            result[product] = orders
            # Info
            # print(f"Order depth for {product}, Buy = ", order_depth.buy_orders, " Sell = ", order_depth.sell_orders)
            if product in own_trades:
                print("Own trade = ", own_trades)
            print("orders placed = ", orders)
            print("Position = ", state.position)
            print("=======================================================================")


        return result