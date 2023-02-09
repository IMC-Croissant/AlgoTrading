from typing import Dict, List
from datamodel import OrderDepth, TradingState, Order
import math

class Trader:

    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        """
        Only method required. It takes all buy and sell orders for all symbols as an input,
        and outputs a list of orders to be sent
        """
        # Initialize the method output dict as an empty dict
        result = {}

        # Iterate over all the keys (the available products) contained in the order depths
        for product in state.order_depths.keys():

            # Run for all products

            # Retrieve the Order Depth containing all the market BUY and SELL orders for PEARLS
            order_depth: OrderDepth = state.order_depths[product]

            # Initialize the list of Orders to be sent as an empty list
            orders: list[Order] = []

            # Define a fair value for the PEARLS.
            # Note that this value of 1000 which is intentionaly high, so that first buy order can exceute
            acceptable_price = 10000
            own_trades = state.own_trades
                
                # 
            print("state = ", state)
            print("state.own_trades = ", state.own_trades)
            print("state.observation = ", state.observations)

            try:
                if len(order_depth.sell_orders) > 0:
                  trades = own_trades[product]
                  acceptable_buy_price = min([trades[i].price for i in range(len(trades)) if trades[i].buyer == "SUBMISSION"])
                        

                  if best_ask < acceptable_buy_price:

                      # Calulating best ask price
                      best_ask = min(order_depth.sell_orders.keys())
                      best_ask_volume = order_depth.sell_orders[best_ask]

                      print("BUY", str(-best_ask_volume) + "x", best_ask)
                      orders.append(Order(product, best_ask, -best_ask_volume))

                if len(order_depth.buy_orders) > 0:

                  # calculating minimum selling price per share (i.e total amount of shares / no. of shares)
                  acceptable_sell_price = sum([trades[i].price * trades[i].quantity  for i in range(len(trades)) if trades[i].seller== "SUBMISSION"])
                  acceptable_sell_price /= sum([trades[i].quantity for i in range(len(trades)) if trades[i].seller == "SUBMISSION"])
                  acceptable_sell_price = math.ceil(acceptable_sell_price)

                  # Calculating best bid price
                  best_bid = max(order_depth.buy_orders.keys())
                  best_bid_volume = order_depth.buy_orders[best_bid]

                  # if best bid price is higher than accepted sell price, 
                  if best_bid > acceptable_sell_price:
                      print("SELL", str(best_bid_volume) + "x", best_bid)
                      orders.append(Order(product, best_bid, -best_bid_volume))

            except:
                if len(order_depth.sell_orders) > 0:
                    best_ask = min(order_depth.sell_orders.keys())
                    best_ask_volume = order_depth.sell_orders[best_ask]

                    # Check if the lowest ask (sell order) is lower than the above defined fair value
                    if best_ask < acceptable_price:
                        print("BUY", str(-best_ask_volume) + "x", best_ask)
                        orders.append(Order(product, best_ask, -best_ask_volume))


                if len(order_depth.buy_orders) >  0:
                    best_bid = max(order_depth.buy_orders.keys())
                    best_bid_volume = order_depth.buy_orders[best_bid]
                    
                    if best_bid > acceptable_price:
                        print("SELL", str(best_bid_volume) + "x", best_bid)
                        orders.append(Order(product, best_bid, -best_bid_volume))

            # Storing orders 
            result[product] = orders

        return result
