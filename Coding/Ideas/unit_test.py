from datamodel import Listing, OrderDepth, Trade, TradingState
#from MM_momo import Trader
#from main_updated_v9_super_trend import Trader
from main_updated_v7 import Trader
import time

timestamp = 600

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
    'COCONUTS': Listing(
		symbol='COCONUTS',
        product='COCONUTS', 
        denomination='COCONUTS',
	),
    'PINA_COLADAS': Listing(
		symbol = 'PINA_COLADAS', 
        product='PINA_COLADAS', 
        denomination='PINA_COLADAS',
	),
    'DIVING_GEAR': Listing(
		symbol = 'DIVING_GEAR', 
        product='DIVING_GEAR', 
        denomination='DIVING_GEAR',
	)
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
    "COCONUTS": OrderDepth(
		buy_orders={142: 3, 141: 5},
		sell_orders={144: -5, 145: -8}
	),
    "PINA_COLADAS": OrderDepth(
		buy_orders={142: 3, 141: 5},
		sell_orders={144: -5, 145: -8}
	),
    "DIVING_GEAR": OrderDepth(
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

observations = {'DOLPHIN_SIGHTINGS': 1200}

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
