# GROUP Croissant - IMC Prosperity Challenge

This repo has all experimental python scripts for the IMC 2023 Challenge.

## Intructions for unit test
The `unit_test.py` file executes a single timestamp locally. 
This is used for the sole purpose to check for typing errors and miscellaneous problems while testing the script.

Currently, in order to run it, is required to execute it within the `Coding/Ideas/.` folder. It takes the trader class
from `main_from_scratch.py`. Other versions can be used.

## Summary of different traders with PnL 

* `MM_momo.py`: added `order_depths`, havent optimized pearls cross strategy yet, need to optimize bananas sitting at 6.33k
* `main_updated.py`: Refactores version of old `MM_momo.py`. Based on *bullish* indicator with SMA 5, 15, 40, 90. Sitting at 6.7k
* `main_updated_v2.py`: Experimental version of `main_updated.py` for long term SMA indicator. Sitting at 6.53k
* `main_updated_v3.py`: Modified version from `main_updated.py` but using EWM instead of SMA, to be optimized. Sitting at 6.65k 
* `main_updated_v4.py`: Includes EWM 5 for price prediction and MACD for signaling (BANANAS), while SMA 90 for PEARLS. Sitting at 4.351K

