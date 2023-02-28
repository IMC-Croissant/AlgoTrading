from typing import Dict, List
from datamodel import OrderDepth, TradingState, Order
import math

class Trader:

## Make a market algo ## 
    def make_a_market(self, l1_bid: int, l1_ask: int) -> tuple:
        spread = l1_ask-l1_bid
        if l1_ask - l1_bid > 6:
            mm_bid = l1_bid + spread*0.1
            mm_ask = l1_ask - spread*0.1
        else:
            mm_bid = 0
            mm_ask = 1000000
        return mm_bid, mm_ask
    
## SMA ## 

## HI

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

        # Iterate over all the keys (the available products) contained in the order depths
        for product in state.order_depths.keys():

            # Retrieve the Order Depth containing all the market BUY and SELL orders for PEARLS
            order_depth: OrderDepth = state.order_depths[product]

            # Initialize the list of Orders to be sent as an empty list
            orders: list[Order] = []

            # print("state = ", state)
            print("state.own_trades = ", own_trades)
            print("state.market_trades = ", market_trades)
            # print("state.observation = ", state.observations)
            print("state.position = ", position)

            level_one_ask = min(order_depth.sell_orders.keys())
            level_one_bid = max(order_depth.buy_orders.keys())
            # Fair Value is the Volume Weighted Average Price
            # mid_price = (level_one_ask + level_one_bid)/2
            # max_long = 20 - position[product] # cannot buy more than this
            # max_short = -20 - position[product] # cannot short more than this

            # ask and bid
            l1_ask = min(order_depth.sell_orders.keys())
            l1_bid = max(order_depth.buy_orders.keys())

            mm_bid, mm_ask = self.make_a_market(l1_bid, l1_ask)

            mm_bid = math.ceil(mm_bid)
            mm_ask = math.floor(mm_ask)
            # volume               
            quantity_ask = order_depth.sell_orders[l1_ask]
            quantity_bid = order_depth.buy_orders[l1_bid]
            quantity = min(quantity_ask, quantity_bid)
       
            orders.append(Order(product, mm_bid, 3))
            orders.append(Order(product, mm_ask, -1*3))
            

            result[product] = orders
            print("orders placed = ", orders)

        return result