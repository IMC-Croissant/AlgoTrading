from typing import Dict, List, Tuple
from datamodel import OrderDepth, TradingState, Order
import pandas as pd
from pandas import DataFrame
import math
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

class Trader:
    ## -- Curr position and amount -- ##
    curr_avg_price = {
        "BANANAS": {
            "average_price": 0,
            "quantity": 0
        },
        "PEARLS": {
            "average_price": 0,
            "quantity": 0
        }
    } # Store average price and quantity for each product at every iteration

    curr_timestamp = 0
    # concatenate information for 0th timestamp
    previous_position = {
        "BANANAS": 0,
        "PEARLS": 0
    }
    # count number of times quantity != position for a prduct
    times_position_not_matched = 0

    # History for trades
    products_ = ['PEARLS', 'BANANAS']
    init_ts = np.zeros(len(products_))
    init_product = products_
    init_mid_prices = [10000, 4948]
    init_avg_cost = [0, 0]
    data = list(zip(init_ts, init_product, init_mid_prices, init_avg_cost))
    # data = list(zip([-1, -1], init_product, init_mid_prices, init_avg_cost))
    _history = pd.DataFrame(data, columns=['timestamp', 'product', 'mid_prices', 'avg_cost'])
    _df_history = pd.DataFrame(list(zip(init_ts, init_product, init_mid_prices,
                                        init_avg_cost, [0, 0], [0,0], [0,0],
                                        [0,0])),
                               columns=['timestamp', 'product', 'prev_mid_price',
                                        'curr_position', 'avg_cost', 'sma_5',
                                        'sma_20', 'sma_50'])


    def get_average_price(self, state: TradingState, product: str) -> float:
        """
        Get average price for a product
        :param state:
        :param product:
        :return:
        """
        if product in state.own_trades:
            # Getting current avarge price and quantity and new price and quantity
            own_trades = state.own_trades[product]
            # more than one order possible for different prices and quantity -> maybe long and sort at the same time
            price_times_qunatity = sum([own_trades[i].price * own_trades[i].quantity
                                        if own_trades[i].buyer == 'SUBMISSION'
                                        else -1 * own_trades[i].quantity * own_trades[i].price
                                        for i in range(len(own_trades))])
            # quantity = sum([own_trades[i].quantity for i in range(len(own_trades))])
            quantity = sum(
                [own_trades[i].quantity
                 if own_trades[i].buyer == 'SUBMISSION'
                 else -1 * own_trades[i].quantity
                 for i in range(len(own_trades))])
            # if quantity < 0:
            #     price_times_qunatity *= -1
            curr_price = self.curr_avg_price[product]["average_price"]  # current average price
            inventory = self.curr_avg_price[product]["quantity"]  # inventory

            # if position changed -> new trade
            if state.position[product] != self.previous_position[product]:
                if inventory + quantity == 0:
                    new_quantity = 0
                    new_avg_price = 0
                else:
                    # Check if position is changing (i.e if sign of position is changing -- from positive to negative or visa versa)
                    postition_change = (inventory > 0) and (quantity > 0)

                    # If position is not changing -- total price invested / total quantity obtained
                    if not postition_change:
                        new_avg_price = (curr_price * inventory + price_times_qunatity) / (inventory + quantity)
                        new_quantity = inventory + quantity

                    # If position is changing -- take quantity with higher abs value and total quantity
                    else:
                        if abs(inventory) > abs(quantity):
                            new_avg_price = curr_price
                            new_quantity = inventory + quantity
                        else:
                            new_avg_price = price_times_qunatity / quantity
                            new_quantity = inventory + quantity

                # modifing curr_avg_price with new values
                print("============== NEW TRADE ===============")
                self.curr_avg_price[product]["average_price"] = new_avg_price
                self.curr_avg_price[product]["quantity"] = new_quantity
                self.previous_position[product] = state.position[product]
                if state.position[product] != new_quantity:
                    self.times_position_not_matched += 1
                    print("-------------POSITION INCOSISTENCY-------------")
                return new_avg_price if new_quantity > 0 else -1 * new_avg_price

        # Else
        return self.curr_avg_price[product]["average_price"] if self.curr_avg_price[product]["quantity"] > 0 else -1 * self.curr_avg_price[product]["average_price"]
        # return self.curr_avg_price[product]["average_price"]

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

        spread = l1_ask - l1_bid
        if product == 'PEARLS':
            thre = 3
            momoFlag = -1

        if product == 'BANANAS':
            thre = 2
            momoFlag = -1

        if spread > thre:
            if momoFlag == 1:  # Bullish Trend --> aggresive bids
                mm_bid = l1_bid + 1
                mm_ask = l1_ask
            elif momoFlag == 0:  # Bearish Trend -> aggressive ask
                mm_bid = l1_bid
                mm_ask = l1_ask - 1
            elif momoFlag == -1:  # No trend -> L1 bid and ask
                mm_bid = l1_bid
                mm_ask = l1_ask
        elif product == 'PEARLS' and spread <= thre:  # liquid market with FV cross
            if l3_bid > 10000:
                mm_ask = l3_bid  # cross the book (sell above FV)
            elif l2_bid > 10000:
                mm_ask = l2_bid
            elif l1_bid > 10000:
                mm_ask = l1_bid
            elif l3_ask < 10000:
                mm_bid = l3_ask  # cross the book (buy below FV)
            elif l2_ask < 10000:
                mm_bid = l2_ask
            elif l1_ask < 10000:
                mm_bid = l1_ask
        elif product == 'BANANAS' and spread <= thre:  # Cross the book but lets look at
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

        buy_quantity = min(max_long, 20)  # max_long can be > 20 we dont ever want to
        sell_quantity = max(max_short, -20)

        # For trendy product we adjust quantity
        if product == 'BANANAS':
            if momo_flag == 1:  # bull trend, so we want to buy more than we sell
                sell_quantity = sell_quantity - 1
                buy_quantity = buy_quantity
            elif momo_flag == 0:  # bear trend, so we want to sell more than we buy
                sell_quantity = sell_quantity
                buy_quantity = buy_quantity - 1

            # We safeguard against trend going against us accordingly
            if cur_pos > 12 and momo_flag == 0:  # bear trend
                sell_quantity += 1
                buy_quantity
            if cur_pos < -12 and momo_flag == 1:  # bull trend
                sell_quantity
                buy_quantity += 1

            return buy_quantity, sell_quantity
        elif product == 'PEARLS':
            return buy_quantity, sell_quantity



    ## -- INVENTORY MANAGER -- ##
    # def _get_cost_of_inventory(self, state: TradingState, product: str) -> float:
    #     history_product = self._df_history[product]

    def _get_cumavg(self, state: TradingState, product: str) -> float:
        df_temp = self._history[self._history['product'] == product]
        history_product = df_temp.set_index('timestamp')
        if state.timestamp > 400:
            cum_avg = history_product.rolling(window=len(history_product)).mean().loc[
                state.timestamp, 'mid_prices']  # we record cumulative avg of price
        else:
            cum_avg = -1
        return cum_avg

    ## SMA ##
    def _get_sma(self, state: TradingState, product: str) -> Tuple[float, float, float]:
        """
        Return SMA 5, 20, 50 for a given product, in form [sma_5, sma_20, sma_50].
        If the timestamp hasn't passed threshold, return -1 for that SMA.
        """
        df_temp = self._history[self._history['product'] == product]
        history_product = df_temp.set_index('timestamp')
        sma_5 = -1
        sma_20 = -1
        sma_50 = -1
        if state.timestamp > 500:
            sma_5 = history_product.rolling(window=5).mean().loc[state.timestamp, 'mid_prices']
        elif state.timestamp > 2000:
            sma_20 = history_product.rolling(window=20).mean().loc[state.timestamp, 'mid_prices']
        elif state.timestamp > 5000:
            sma_50 = history_product.rolling(window=50).mean().loc[state.timestamp, 'mid_prices']
        return sma_5, sma_20, sma_50

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

    # def _get_avg_cost(self, state: TradingState) -> list:

    #     signed_quant = [] # 2d array of signed quantities [ [2, 1], [-1] ]
    #     price_per_quant = [] # same dimension as signed quant with respective prices

    #     for prod in state.order_depths.keys():
    #         signed_quant.append([state.own_trades[prod][x].quantity if state.own_trades[prod][x].buyer == 'SUBMISSION' else -1*state.own_trades[prod][x].quantity for x in range(len(state.own_trades[prod]))])
    #         price_per_quant.append([state.own_trades[prod][x].price for x in range(len(state.own_trades[prod]))])

    #     avg = [] # the list of avg cost
    #     quantity = [np.sum(signed_quant[i]) for i in range(len(signed_quant))]

    #     for i in range(len(price_per_quant)):
    #         avg.append(np.dot(price_per_quant[i], signed_quant[i])/abs(sum(signed_quant[i])))
    #     return avg, quantity

    ## HISTORICAL FOR SMA COMPUTATION ##
    def _process_new_data(self, state: TradingState, products) -> None:
        """Adds new data point to historical data."""

        timestamp = np.ones(len(products)) * state.timestamp  # time stamp for each product

        mid_prices = []  # mid prices
        avg_cost = []  # avg cost
        for key in state.order_depths.keys():
            l1_ask = min(state.order_depths[key].sell_orders)
            l1_bid = max(state.order_depths[key].buy_orders)
            cur_mid = (l1_ask + l1_bid) / 2
            mid_prices.append(cur_mid)
            avg_cost.append(self.get_average_price(state, key))

        # avg_cost = self.get_average_price(state, products) # avg costs

        our_columns = ['timestamp', 'product', 'mid_prices', 'avg_cost']

        data = list(zip(timestamp, products, mid_prices, avg_cost))

        temp_df = pd.DataFrame(data, columns=our_columns)

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
        products = [prod for prod in state.order_depths.keys()]
        # Store data from current timestamp
        self._process_new_data(state, products)

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
            max_long = 20 - cur_pos  # cannot buy more than this
            max_short = -20 - cur_pos  # cannot short more than this

            # Momentum Flag: if 20SMA >< 50SMA inventory & spreads change accordingly. Use HURST exp to ID if trendy
            sma_5, sma_20, sma_50 = self._get_sma(state, product)
            cum_avg = self._get_cumavg(state, product)

            if sma_50 == -1:
                momo_flag = -1  # Not enough history just make regular market
            elif sma_50 < sma_20:
                momo_flag = 1  # Bullish Trend
            else:
                momo_flag = 0  # Bearish Trend

            if product == 'PEARLS':
                momo_flag = -1

            # LEVEL 1: ask and bid
            asks = sorted(order_depth.sell_orders.keys())
            bids = sorted(order_depth.buy_orders.keys())

            fairvalue = sma_5  # this is always our fair value

            if cum_avg == -1:
                fairvalue = 4928
            elif sma_20 == -1:
                fairvalue = cum_avg  # let fair value be the avg of the first couple days
            else:
                fairvalue == sma_20

            # MAKE THE MARKET
            mm_bid, mm_ask = self.make_a_market(asks, bids, momo_flag, product, fairvalue)  # assign our bid/ask spread

            # INVENTORY MANAGEMENT/VOLUME
            buy_quantity, sell_quantity = self.get_quantity(product, max_long, max_short, cur_pos, momo_flag)
            buy_quantity = min(buy_quantity, 20)  # max_long can be > 20 we dont ever want to
            sell_quantity = max(sell_quantity, -20)
            # ORDER UP!
            # if product == 'PEARLS'/'BANANAS':
            orders.append(Order(product, mm_bid, buy_quantity))
            orders.append(Order(product, mm_ask, sell_quantity))

            result[product] = orders

            print("state.own_trades = ", own_trades)
            print("state.market_trades = ", market_trades)
            print("state.position = ", position)
            if state.timestamp == 1999 * 100:
                print(f"{self.times_position_not_matched} times quantity and position not matched")
                print("\n Here is the final DataFrame \n")
                print(self._history)
            print("orders placed = ", orders)
            print("Avg price = ", self.get_average_price(state=state, product=product))
            print("Current position amount = ", self.curr_avg_price)
            print("------------------------------------------------------------------------------------------\n")

        return result
