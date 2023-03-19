from datamodel import Listing, OrderDepth, Trade, TradingState
#from MM_momo import Trader
from main_update_v5 import Trader
import time

timestamp = 60000

listings = {
	"PEARLS": Listing(
		symbol="PEARLS",
		product="PEARLS",
		denomination= "SEASHELLS",
	),
	"BANANAS": Listing(
		symbol="BANANAS",
		product="BANANAS",
		denomination="SEASHELLS",
	),
}

order_depths = {
	"PEARLS": OrderDepth(
		buy_orders={10: 7, 9: 5},
		sell_orders={20: -5, 13: -3}
	),
	"BANANAS": OrderDepth(
		buy_orders={142: 3, 141: 5},
		sell_orders={144: -5, 145: -8}
	),
}

own_trades = {
	"PEARLS": [
		Trade(
			symbol="PEARLS",
			price=11,
			quantity=4,
			buyer="SUBMISSION",
			seller="",
		),
		Trade(
			symbol="PEARLS",
			price=12,
			quantity=3,
			buyer="SUBMISSION",
			seller="",
		)
	],
	"BANANAS": [
		Trade(
			symbol="BANANAS",
			price=143,
			quantity=2,
			buyer="",
			seller="SUBMISSION",
		),
	]
}

market_trades = {
	"PEARLS": [],
	"BANANAS": []
}

position = {
	"PEARLS": 3,
	"BANANAS": 0
}

observations = {}

state = TradingState(
	timestamp,
        listings,
	order_depths,
	own_trades,
	market_trades,
        position,
        observations,
)

start_time = time.time()
trader = Trader()
trader.run(state)
elapsed_time = time.time() - start_time
print("total time for an iteration {} [s]".format(elapsed_time))
