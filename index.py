from fastapi import FastAPI
from nselib import capital_market
import yfinance as yf
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from pydantic import BaseModel
import concurrent.futures
import logging
import redis
import os
import json
import time
import traceback

logging.basicConfig(level=logging.INFO)

redis_client = redis.Redis(host='localhost', port=6379, db=0)

# Now use this session to access NSE data
app = FastAPI()

@app.get("/equity-list")
def get_nse_data():
    # Fetch data from NSE, for example: nifty index data
    try:
        nifty_data = capital_market.equity_list()  # Or any other method from nselib
        if isinstance(nifty_data, pd.DataFrame):
            # Convert DataFrame to dictionary
            nifty_data_dict = nifty_data.to_dict(orient="records")  # You can use other orientations
            return JSONResponse(content=nifty_data_dict, status_code=200)
        else:
            return JSONResponse(content={"error": "Unexpected data format"}, status_code=500)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/equity/{sym}")
def get_equity_info(sym):
    try:
        print(sym)
        time.sleep(2)  # wait 2 seconds before each call
        ticker = yf.Ticker(sym)  # Use the provided symbol instead of hardcoded "AAPL"
        print(ticker)
        nifty_data = ticker.info  # Access the info attribute directly
        
        if isinstance(nifty_data, dict):
            return JSONResponse(content=nifty_data, status_code=200)  # Return the dictionary as is
        else:
            return JSONResponse(content={"error": "Unexpected data format"}, status_code=500)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/indices-list")
def get_all_indices():
    # Fetch data from NSE, for example: nifty index data
    try:
        nifty_data = capital_market.market_watch_all_indices()  # Or any other method from nselib
        # Check if the data is valid
        if nifty_data is None or len(nifty_data) == 0:
            return JSONResponse(content={"error": "No data returned from NSE"}, status_code=500)
        
        # Ensure the data is a DataFrame
        if not isinstance(nifty_data, pd.DataFrame):
            return JSONResponse(content={"error": "Unexpected data format from NSE"}, status_code=500)
        
        # Clean and process the data as before
        nifty_data_clean = nifty_data.replace([np.nan, np.inf, -np.inf], None)
        
        if 'key' in nifty_data_clean.columns:
            sectoral_indices = nifty_data_clean[nifty_data_clean['key'] == 'SECTORAL INDICES']
        else:
            return JSONResponse(content={"error": "'key' column not found"}, status_code=500)
        
        if 'percentChange' in sectoral_indices.columns:
            nifty_data_sorted = sectoral_indices.sort_values(by='percentChange', ascending=False)
        elif 'variation' in sectoral_indices.columns:
            nifty_data_sorted = sectoral_indices.sort_values(by='variation', ascending=False)
        else:
            return JSONResponse(content={"error": "No valid change column for sorting."}, status_code=500)
        
        nifty_data_dict = nifty_data_sorted.to_dict(orient="records")
        return JSONResponse(content=nifty_data_dict, status_code=200)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    
@app.get('/index/{index}')
def get_index_data(index, from_date = "20-09-2023", to_date = "03-10-2024"):
    try:
        # Fetch index data (make sure this function returns a valid structure)
        index_data = capital_market.index_data(index, from_date=from_date, to_date=to_date)
        if isinstance(index_data, pd.DataFrame):
            index_data_clean = index_data.replace([np.nan, np.inf, -np.inf], None)
        
        index_data_dict = index_data_clean.to_dict(orient="records")

        return JSONResponse(content=index_data_dict, status_code=200)

    except Exception as e:
        # Raise a 404 error for not found or return a generic error message
        return JSONResponse(content={"error": str(e)}, status_code=500)
    
@app.get("/deliveries/{sym}/{period}")
def get_deliveries(sym: str, period: str):
    try:
        # Fetch data from NSE using the price_volume_and_deliverable_position_data function
        nifty_data = capital_market.deliverable_position_data(sym, period=period)
        
        if isinstance(nifty_data, pd.DataFrame):
            # Replace NaN and infinite values with None
            nifty_data_clean = nifty_data.replace([np.nan, np.inf, -np.inf], None)
            
            # Convert DataFrame to dictionary
            nifty_data_dict = nifty_data_clean.to_dict(orient="records")
            
            # Use jsonable_encoder to ensure it's JSON-serializable
            return jsonable_encoder(nifty_data_dict)
        else:
            return {"error": "Unexpected data format"}
    except Exception as e:
        return {"error": str(e)}
    
def calculate_support_resistance(hist):
    # Support is the lowest Close price over the period; Resistance is the highest Close price
    support = hist['Close'].min()
    resistance = hist['Close'].max()
    return support, resistance

# Function to calculate RSI
def calculate_rsi(hist, period=14):
    delta = hist['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Function to calculate MACD
def calculate_macd(hist):
    exp1 = hist['Close'].ewm(span=12, adjust=False).mean()
    exp2 = hist['Close'].ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal

# Function to calculate Bollinger Bands
def calculate_bollinger_bands(hist, window=20, num_std_dev=2):
    rolling_mean = hist['Close'].rolling(window=window).mean()
    rolling_std = hist['Close'].rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_std_dev)
    lower_band = rolling_mean - (rolling_std * num_std_dev)
    return upper_band, lower_band

# def generate_signals(row, avg_10_days):
#     # Initialize signal counters
#     buy_signals = 0
#     sell_signals = 0
#     neutral_signals = 0

#     # Weights for indicators
#     weights = {
#         '50_200_MA_Crossover': 1, 
#         '50_200_MA_Crossover_Status': "Neutral",
#         'MACD': 1, 
#         'MACD_Status': "Neutral",
#         'RSI': 1,
#         'RSI_Status': "Neutral",
#         'Bollinger_Bands': 1, 
#         'Bollinger_Bands_Status': "Neutral",
#         'Delivery': 1,
#         'Delivery_Status': "Neutral",
#         'Volume': 1,
#         'Volume_Status': "Neutral",
#         "50_MA": 1,
#         '50_MA_Status': "Neutral",
#         "Delivery_Volume": 1,
#         "Delivery_Volume_Status": "Neutral",
#         "VWAP": 1,
#         "VWAP_Status": "Neutral",
#         # "ROC": 1,  # Reinserting ROC with a weight
#         # 'ROC_Status': "Neutral"
#     }

#     total_indicators = 9  # Now includes ROC

#     # Indicator 1: 50-day and 200-day MA Crossover
#     ma_crossover_strength = row['50_MA'] - row['200_MA']
#     if row['50_MA'] > row['200_MA'] and row['Prev_50_MA'] <= row['Prev_200_MA']:
#         buy_signals += weights['50_200_MA_Crossover'] if ma_crossover_strength > 0.5 else 0.5 * weights['50_200_MA_Crossover']
#         weights['50_200_MA_Crossover_Status'] = "Positive"
#     elif row['50_MA'] < row['200_MA'] and row['Prev_50_MA'] >= row['Prev_200_MA']:
#         sell_signals += weights['50_200_MA_Crossover'] if ma_crossover_strength < -0.5 else 0.5 * weights['50_200_MA_Crossover']
#         weights['50_200_MA_Crossover_Status'] = "Negative"
#     else:
#         neutral_signals += 1

#     # Indicator 2: RSI conditions
#     if row['RSI'] < 30:
#         buy_signals += weights['RSI']
#         weights['RSI_Status'] = "Positive"
#     elif row['RSI'] > 70:
#         sell_signals += weights['RSI']
#         weights['RSI_Status'] = "Negative"
#     else:
#         neutral_signals += 1

#     # Indicator 3: MACD conditions
#     if row['MACD'] > row['MACD_Signal']:
#         buy_signals += weights['MACD']
#         weights['MACD_Status'] = "Positive"
#     elif row['MACD'] < row['MACD_Signal']:
#         sell_signals += weights['MACD']
#         weights['MACD_Status'] = "Negative"
#     else:
#         neutral_signals += 1

#     # Indicator 4: Delivery percentage conditions
#     if (avg_10_days < row['%DlyQttoTradedQty'] or row['%DlyQttoTradedQty'] > 50):
#         buy_signals += weights['Delivery']
#         weights['Delivery_Status'] = "Positive"
#     else:
#         sell_signals += weights['Delivery']
#         weights['Delivery_Status'] = "Negative"

#     # Indicator 5: Volume conditions
#     if row['Volume'] > row['10_MA_Volume']:
#         buy_signals += weights['Volume']
#         weights['Volume_Status'] = "Positive"
#     else:
#         sell_signals += weights['Volume']
#         weights['Volume_Status'] = "Negative"

#     # Indicator 6: Bollinger Bands
#     distance_from_upper = row['Close'] - row['Upper_Band']
#     distance_from_lower = row['Close'] - row['Lower_Band']
#     if distance_from_lower < 0 and abs(distance_from_lower) > 1.5 * (row['Upper_Band'] - row['Lower_Band']):
#         buy_signals += weights['Bollinger_Bands']
#         weights['Bollinger_Bands_Status'] = "Positive"
#     elif distance_from_upper > 0 and abs(distance_from_upper) > 1.5 * (row['Upper_Band'] - row['Lower_Band']):
#         sell_signals += weights['Bollinger_Bands']
#         weights['Bollinger_Bands_Status'] = "Negative"
#     else:
#         neutral_signals += 1

#     # Indicator 7: 50-day Moving Average
#     if row['Close'] > row['50_MA']:
#         buy_signals += weights['50_MA']
#         weights['50_MA_Status'] = "Positive"
#     elif row['Close'] < row['50_MA']:
#         sell_signals += weights['50_MA']
#         weights['50_MA_Status'] = "Negative"
#     else:
#         neutral_signals += 1

#     # Indicator 8: VWAP condition
#     if row['Close'] < row['VWAP']:
#         buy_signals += weights['VWAP']
#         weights['VWAP_Status'] = "Positive"
#     elif row['Close'] > row['VWAP']:
#         sell_signals += weights['VWAP']
#         weights['VWAP_Status'] = "Negative"
#     else:
#         neutral_signals += 1

#     # Indicator 9: Delivery and Volume
#     if (avg_10_days < row['%DlyQttoTradedQty'] or row['%DlyQttoTradedQty'] > 50) and row['Volume'] > row['10_MA_Volume']:
#         buy_signals += weights['Delivery_Volume']
#         weights['Delivery_Volume_Status'] = "Positive"
#     else:
#         sell_signals += weights['Delivery_Volume']
#         weights['Delivery_Volume_Status'] = "Negative"

#     # # Indicator 10: ROC (Rate of Change) condition
#     # roc_strength = row['ROC']  # ROC can be dynamic based on time periods
#     # if roc_strength > 0:
#     #     buy_signals += weights['ROC']
#     #     weights['ROC_Status'] = "Positive"
#     # elif roc_strength < 0:
#     #     sell_signals += weights['ROC']
#     #     weights['ROC_Status'] = "Negative"
#     # else:
#     #     neutral_signals += 1

#     # Adjust buy/sell signal strength considering neutral signals
#     active_indicators = total_indicators - neutral_signals

#     # Prevent division by zero
#     if active_indicators > 0:
#         signal_strength = (buy_signals / active_indicators) * 100
#         signal_strength_sell = (sell_signals / active_indicators) * 100
#     else:
#         signal_strength = 0
#         signal_strength_sell = 0

#     # Decision logic based on the accumulated signals
#     if signal_strength > 55:  # Adjust threshold for stronger buy signals
#         return {
#             "weights": {k: v for k, v in weights.items() if "_Status" in k},
#             "signalStrength": f"{signal_strength:.2f}%",
#             "buyValue": buy_signals,
#             "sellValue": sell_signals,
#             "neutralValue": neutral_signals,
#             "signal": "Buy"
#         }
#     elif signal_strength_sell > 55:  # Adjust threshold for stronger sell signals
#         return {
#             "weights": {k: v for k, v in weights.items() if "_Status" in k},
#             "signalStrength": f"{signal_strength_sell:.2f}%",
#             "buyValue": buy_signals,
#             "sellValue": sell_signals,
#             "neutralValue": neutral_signals,
#             "signal": "Sell"
#         }
#     else:
#         return {
#             "weights": {k: v for k, v in weights.items() if "_Status" in k},
#             "signalStrength": "Neutral",
#             "buyValue": buy_signals,
#             "sellValue": sell_signals,
#             "neutralValue": neutral_signals,
#             "signal": "Hold"
#         }

# def generate_signals(row, avg_10_days):
    buy_signals = 0
    sell_signals = 0
    active_indicators = 0

    # Indicator weights
    weights = {
        'MA_Crossover': 1,
        'RSI': 1,
        'MACD': 1,
        'Delivery': 1,
        'Volume': 1,
        'Bollinger': 1,
        'MA50_Close': 1,
        'VWAP': 1,
        'Delivery_Volume': 1,
        # 'ROC': 1,  # Uncomment if ROC is used
    }

    statuses = {f"{k}_Status": "Neutral" for k in weights}

    # 1. 50/200 MA crossover
    if row['50_MA'] > row['200_MA'] and row['Prev_50_MA'] <= row['Prev_200_MA']:
        buy_signals += weights['MA_Crossover']
        active_indicators += 1
        statuses['MA_Crossover_Status'] = "Positive"
    elif row['50_MA'] < row['200_MA'] and row['Prev_50_MA'] >= row['Prev_200_MA']:
        sell_signals += weights['MA_Crossover']
        active_indicators += 1
        statuses['MA_Crossover_Status'] = "Negative"

    # 2. RSI
    if row['RSI'] < 30:
        buy_signals += weights['RSI']
        active_indicators += 1
        statuses['RSI_Status'] = "Positive"
    elif row['RSI'] > 70:
        sell_signals += weights['RSI']
        active_indicators += 1
        statuses['RSI_Status'] = "Negative"

    # 3. MACD
    if row['MACD'] > row['MACD_Signal']:
        buy_signals += weights['MACD']
        active_indicators += 1
        statuses['MACD_Status'] = "Positive"
    elif row['MACD'] < row['MACD_Signal']:
        sell_signals += weights['MACD']
        active_indicators += 1
        statuses['MACD_Status'] = "Negative"

    # 4. Delivery %
    if row['%DlyQttoTradedQty'] > 50 or row['%DlyQttoTradedQty'] > avg_10_days:
        buy_signals += weights['Delivery']
        active_indicators += 1
        statuses['Delivery_Status'] = "Positive"
    else:
        sell_signals += weights['Delivery']
        active_indicators += 1
        statuses['Delivery_Status'] = "Negative"

    # 5. Volume vs 10-day avg
    if row['Volume'] > row['10_MA_Volume']:
        buy_signals += weights['Volume']
        active_indicators += 1
        statuses['Volume_Status'] = "Positive"
    else:
        sell_signals += weights['Volume']
        active_indicators += 1
        statuses['Volume_Status'] = "Negative"

    # 6. Bollinger Band
    bb_range = row['Upper_Band'] - row['Lower_Band']
    if row['Close'] < row['Lower_Band'] - 0.5 * bb_range:
        buy_signals += weights['Bollinger']
        active_indicators += 1
        statuses['Bollinger_Status'] = "Positive"
    elif row['Close'] > row['Upper_Band'] + 0.5 * bb_range:
        sell_signals += weights['Bollinger']
        active_indicators += 1
        statuses['Bollinger_Status'] = "Negative"

    # 7. Close vs 50 MA
    if row['Close'] > row['50_MA']:
        buy_signals += weights['MA50_Close']
        active_indicators += 1
        statuses['MA50_Close_Status'] = "Positive"
    elif row['Close'] < row['50_MA']:
        sell_signals += weights['MA50_Close']
        active_indicators += 1
        statuses['MA50_Close_Status'] = "Negative"

    # 8. VWAP
    if row['Close'] < row['VWAP']:
        buy_signals += weights['VWAP']
        active_indicators += 1
        statuses['VWAP_Status'] = "Positive"
    elif row['Close'] > row['VWAP']:
        sell_signals += weights['VWAP']
        active_indicators += 1
        statuses['VWAP_Status'] = "Negative"

    # 9. Combined Delivery + Volume
    if (row['%DlyQttoTradedQty'] > avg_10_days or row['%DlyQttoTradedQty'] > 50) and row['Volume'] > row['10_MA_Volume']:
        buy_signals += weights['Delivery_Volume']
        active_indicators += 1
        statuses['Delivery_Volume_Status'] = "Positive"
    else:
        sell_signals += weights['Delivery_Volume']
        active_indicators += 1
        statuses['Delivery_Volume_Status'] = "Negative"

    # 10. Optional: ROC
    # if 'ROC' in row:
    #     if row['ROC'] > 0:
    #         buy_signals += weights['ROC']
    #         active_indicators += 1
    #         statuses['ROC_Status'] = "Positive"
    #     elif row['ROC'] < 0:
    #         sell_signals += weights['ROC']
    #         active_indicators += 1
    #         statuses['ROC_Status'] = "Negative"

    # Compute signal strength
    if active_indicators > 0:
        buy_strength = (buy_signals / active_indicators) * 100
        sell_strength = (sell_signals / active_indicators) * 100
    else:
        buy_strength = sell_strength = 0

    print("Buy: ", buy_strength, active_indicators)
    print("Sell :", sell_strength)
    # Decision
    if buy_strength > 50:
        signal = "Buy"
        signal_strength = f"{buy_strength:.2f}%"
    elif sell_strength > 50:
        signal = "Sell"
        signal_strength = f"{sell_strength:.2f}%"
    else:
        signal = "Hold"
        signal_strength = "Neutral"

    return {
        "signal": signal,
        "signalStrength": signal_strength,
        "buyValue": buy_signals,
        "sellValue": sell_signals,
        "neutralValue": len(weights) - active_indicators,
        "weights": statuses
    }

def generate_signals(row, avg_10_days):
    # Initialize signal counters
    buy_signals = 0
    sell_signals = 0
    active_indicators = 0
    confidence_score = 0
    max_confidence = 5

    # Weights for each indicator
    weights = {
        '21_44_MA_Crossover': 1,
        '21_44_MA_Crossover_Status': "Neutral",
        'MACD': 1,
        'MACD_Status': "Neutral",
        'RSI': 1,
        'RSI_Status': "Neutral",
        'Bollinger_Bands': 1,
        'Bollinger_Bands_Status': "Neutral",
        'Delivery': 1,
        'Delivery_Status': "Neutral",
        'Volume': 1,
        'Volume_Status': "Neutral",
        '21_MA': 1,
        '21_MA_Status': "Neutral",
        'VWAP': 1,
        'VWAP_Status': "Neutral",
        'Delivery_Volume': 1,
        'Delivery_Volume_Status': "Neutral",
    }

    # Indicator 1: 21/44 MA Crossover
    ma_crossover_strength = row['21_MA'] - row['44_MA']
    confidence_score += abs(ma_crossover_strength) / max(row['44_MA'], 1) * 10
    if row['21_MA'] > row['44_MA'] and row['Prev_21_MA'] <= row['Prev_44_MA']:
        buy_signals += weights['21_44_MA_Crossover']
        weights['21_44_MA_Crossover_Status'] = "Positive"
        active_indicators += 1
    elif row['21_MA'] < row['44_MA'] and row['Prev_21_MA'] >= row['Prev_44_MA']:
        sell_signals += weights['21_44_MA_Crossover']
        weights['21_44_MA_Crossover_Status'] = "Negative"
        active_indicators += 1

    # Indicator 2: RSI
    if row['RSI'] < 30:
        buy_signals += weights['RSI']
        weights['RSI_Status'] = "Positive"
        active_indicators += 1
    elif row['RSI'] > 70:
        sell_signals += weights['RSI']
        weights['RSI_Status'] = "Negative"
        active_indicators += 1

    confidence_score += abs(50 - row['RSI']) / 50 * 5  # stronger if farther from 50

    # Indicator 3: MACD
    if row['MACD'] > row['MACD_Signal']:
        buy_signals += weights['MACD']
        weights['MACD_Status'] = "Positive"
        active_indicators += 1
    elif row['MACD'] < row['MACD_Signal']:
        sell_signals += weights['MACD']
        weights['MACD_Status'] = "Negative"
        active_indicators += 1

    confidence_score += abs(row['MACD'] - row['MACD_Signal']) / max(abs(row['MACD_Signal']), 0.01) * 5

    # Indicator 4: Delivery %
    if (avg_10_days < row['%DlyQttoTradedQty'] or row['%DlyQttoTradedQty'] > 50):
        buy_signals += weights['Delivery']
        weights['Delivery_Status'] = "Positive"
        active_indicators += 1
    else:
        sell_signals += weights['Delivery']
        weights['Delivery_Status'] = "Negative"
        active_indicators += 1

    # Indicator 5: Volume
    if row['Volume'] > row['10_MA_Volume']:
        buy_signals += weights['Volume']
        weights['Volume_Status'] = "Positive"
        active_indicators += 1
    else:
        sell_signals += weights['Volume']
        weights['Volume_Status'] = "Negative"
        active_indicators += 1

    # Indicator 6: Bollinger Bands
    band_range = row['Upper_Band'] - row['Lower_Band']
    if row['Close'] < row['Lower_Band'] - 0.5 * band_range:
        buy_signals += weights['Bollinger_Bands']
        weights['Bollinger_Bands_Status'] = "Positive"
        confidence_score += (row['Lower_Band'] - row['Close']) / band_range * 5
        active_indicators += 1
    elif row['Close'] > row['Upper_Band'] + 0.5 * band_range:
        sell_signals += weights['Bollinger_Bands']
        weights['Bollinger_Bands_Status'] = "Negative"
        confidence_score += (row['Close'] - row['Upper_Band']) / band_range * 5
        active_indicators += 1

    # Indicator 7: Price vs 21_MA
    if row['Close'] > row['21_MA']:
        buy_signals += weights['21_MA']
        weights['21_MA_Status'] = "Positive"
        active_indicators += 1
    elif row['Close'] < row['21_MA']:
        sell_signals += weights['21_MA']
        weights['21_MA_Status'] = "Negative"
        active_indicators += 1

    # Indicator 8: VWAP
    if row['Close'] < row['VWAP']:
        buy_signals += weights['VWAP']
        weights['VWAP_Status'] = "Positive"
        active_indicators += 1
    elif row['Close'] > row['VWAP']:
        sell_signals += weights['VWAP']
        weights['VWAP_Status'] = "Negative"
        active_indicators += 1

    confidence_score += abs(row['Close'] - row['VWAP']) / max(row['VWAP'], 1) * 5

    # Indicator 9: Delivery + Volume combo
    if (avg_10_days < row['%DlyQttoTradedQty'] or row['%DlyQttoTradedQty'] > 50) and row['Volume'] > row['10_MA_Volume']:
        buy_signals += weights['Delivery_Volume']
        weights['Delivery_Volume_Status'] = "Positive"
        active_indicators += 1
    else:
        sell_signals += weights['Delivery_Volume']
        weights['Delivery_Volume_Status'] = "Negative"
        active_indicators += 1

    # Final Signal Strength Calculation
    if active_indicators > 0:
        signal_strength = (buy_signals / active_indicators) * 100
        signal_strength_sell = (sell_signals / active_indicators) * 100
        confidence_score = confidence_score / (active_indicators ** 0.5)  # sqrt to reduce dilution
    else:
        signal_strength = signal_strength_sell = 0

    # Final Decision
    if signal_strength > 55:
        signal = "Buy"
    elif signal_strength_sell > 55:
        signal = "Sell"
    else:
        signal = "Hold"

    return {
        "weights": {k: v for k, v in weights.items() if "_Status" in k},
        "signalStrength": f"{max(signal_strength, signal_strength_sell):.2f}%",
        "buyValue": buy_signals,
        "sellValue": sell_signals,
        "activeIndicators": active_indicators,
        "signal": signal,
        "confidenceScore": min(round(confidence_score, 4) / max_confidence, 1.0),
    }


import logging


def is_sideways_market(hist, threshold=0.02, volatility_threshold_factor=0.5):
    """
    Determine if the market is in a sideways trend or trending.
    
    Parameters:
        hist (DataFrame): Historical price data containing 'ClosePrice', 'High', and 'Low'.
        threshold (float): Percentage threshold for determining trend around the SMA.
        volatility_threshold_factor (float): Factor to scale the mean price for volatility check.
    
    Returns:
        bool: True if the market is sideways, False if trending.
    """
    # Calculate the Simple Moving Average (SMA)
    hist['SMA'] = hist['ClosePrice'].rolling(window=20).mean()

    if hist['High'].dtype == 'object':
        hist['High'] = hist['High'].str.replace(',', '').str.strip()

    if hist['Low'].dtype == 'object':
        hist['Low'] = hist['Low'].str.replace(',', '').str.strip()

    # Clean 'High' and 'Low' columns
    hist['High'] = pd.to_numeric(hist['High'], errors='coerce')
    hist['Low'] = pd.to_numeric(hist['Low'], errors='coerce')

    # Calculate True Range (TR) and Average True Range (ATR)
    high_low = hist['High'] - hist['Low']
    high_close = abs(hist['High'] - hist['ClosePrice'].shift(1))
    low_close = abs(hist['Low'] - hist['ClosePrice'].shift(1))
    true_range = pd.DataFrame({'high_low': high_low, 'high_close': high_close, 'low_close': low_close})
    hist['ATR'] = true_range.max(axis=1).rolling(window=14).mean()

    # Determine if price is above or below the SMA with the given threshold
    price_above_sma = hist['ClosePrice'] > hist['SMA'] * (1 + threshold)
    price_below_sma = hist['ClosePrice'] < hist['SMA'] * (1 - threshold)

    # Check if the market is trending
    if price_above_sma.any() or price_below_sma.any():
        return False  # Market is trending

    # Determine volatility condition using ATR
    mean_price = hist['ClosePrice'].mean()
    volatility_threshold = volatility_threshold_factor * mean_price  # Define volatility threshold

    # Check if the ATR indicates low volatility
    if hist['ATR'].iloc[-1] < volatility_threshold:
        return True  # Indicates sideways market due to low volatility
    
    return False  # Indicates market is trending due to high volatility

    
def calculate_date_range(timeframe):
    current_date = datetime.now()
    
    # Handle "mo" for months
    if 'mo' in timeframe:
        unit = 'm'
        amount = int(timeframe.replace('mo', ''))
    else:
        unit = timeframe[-1]  # 'd' for days, 'm' for months, 'y' for years
        amount = int(timeframe[:-1])  # numeric part of the timeframe

    if unit == 'd':
        from_date = current_date - timedelta(days=amount)
    elif unit == 'm':
        from_date = current_date - relativedelta(months=amount)
    elif unit == 'y':
        from_date = current_date - relativedelta(years=amount)
    else:
        raise ValueError("Invalid timeframe format. Use 'd', 'mo', or 'y'.")
    
    return {
        "from_date": from_date.strftime('%d-%m-%Y'), 
        "to_date": current_date.strftime('%d-%m-%Y')
    }


class StockAnalysisResponse(BaseModel):
    symbol: str
    support: float
    resistance: float
    stopLoss: float
    average_daily_qty: float
    signal: str
    signal_strength: float

# def dataAnalysis(sym, period):
#     try: 
#         ySym = sym + ".NS"        
#         customDate = calculate_date_range(period)

#         hist = yf.download(ySym, period=period)
#         hist2 = capital_market.price_volume_and_deliverable_position_data(sym, 
#                 from_date=customDate['from_date'], to_date=customDate['to_date'])

#         if hist.empty:
#             return JSONResponse(content={"error": "No data available. Check the symbol or period."}, status_code=404)
        
#         if hist2.empty:
#             return JSONResponse(content={"error": "No data available. Check the symbol or period."}, status_code=404)

#         # Reset index to get the 'Date' as a column
#         hist = hist.reset_index()

#         # Clean and convert columns to numeric
#         if 'Close' in hist.columns:
#             if hist['Close'].dtype == 'object':
#                 hist['ClosePrice'] = hist['Close'].str.replace(',', '').str.strip()
#             else:
#                 hist['ClosePrice'] = hist['Close']
#         else:
#             return JSONResponse(content={"error": "Close price data not found."}, status_code=404)

#         if 'Volume' in hist.columns:
#             if hist['Volume'].dtype == 'object':
#                 hist['Volume'] = hist['Volume'].str.replace(',', '').str.strip()
#         else:
#             return JSONResponse(content={"error": "Volume data not found."}, status_code=404)

#         # Convert columns to numeric
#         hist['ClosePrice'] = pd.to_numeric(hist['ClosePrice'], errors='coerce')
#         hist['Volume'] = pd.to_numeric(hist['Volume'], errors='coerce')

#         # Forward fill for ClosePrice to avoid NaN issues
#         hist['ClosePrice'].fillna(method='ffill', inplace=True)

#         # Ensure there's valid data in Volume and ClosePrice
#         if hist['Volume'].isnull().all() or hist['ClosePrice'].isnull().all():
#             return JSONResponse(content={"error": "No valid data available for VWAP calculation."}, status_code=404)

#         # VWAP calculation
#         hist['Price_Volume'] = hist['ClosePrice'] * hist['Volume']
#         cumulative_price_volume = hist['Price_Volume'].cumsum()
#         cumulative_volume = hist['Volume'].cumsum()

#         # Calculate VWAP, handle division by zero
#         hist['VWAP'] = np.where(cumulative_volume != 0, cumulative_price_volume / cumulative_volume, 0)

#         # Ensure sufficient data points for moving averages
#         if len(hist) < 44:
#             return JSONResponse(content={"error": "Not enough data points to calculate moving averages."}, status_code=400)

#         # Calculate Moving Averages
#         hist['50_MA'] = hist['ClosePrice'].rolling(window=50).mean()
#         hist['100_MA'] = hist['ClosePrice'].rolling(window=100).mean()
#         hist['200_MA'] = hist['ClosePrice'].rolling(window=200).mean()

#         # Create shifted columns for previous day's MA values
#         hist['Prev_50_MA'] = hist['50_MA'].shift(1)
#         hist['Prev_200_MA'] = hist['200_MA'].shift(1)

#         # Calculate Exponential Moving Averages (EMA)
#         hist['44_EMA'] = hist['ClosePrice'].ewm(span=44, adjust=False).mean()  # 44-day EMA
#         hist['100_EMA'] = hist['ClosePrice'].ewm(span=100, adjust=False).mean()  # 100-day EMA

#         # Calculate 10, 20, 50-day moving average for Volume
#         if len(hist) >= 50:
#             hist['10_MA_Volume'] = hist['Volume'].rolling(window=10).mean()
#             hist['20_MA_Volume'] = hist['Volume'].rolling(window=20).mean()
#             hist['50_MA_Volume'] = hist['Volume'].rolling(window=50).mean()
#         else:
#             hist['10_MA_Volume'] = None
#             hist['20_MA_Volume'] = None
#             hist['50_MA_Volume'] = None

#         # Calculate Support and Resistance Levels
#         support, resistance = calculate_support_resistance(hist)

#         # Calculate Indicators
#         hist['RSI'] = calculate_rsi(hist)
#         hist['MACD'], hist['MACD_Signal'] = calculate_macd(hist)
#         hist['Upper_Band'], hist['Lower_Band'] = calculate_bollinger_bands(hist)

#         # Calculate Stop-Loss Level (5% below the last closing price)
#         last_close = hist['ClosePrice'].iloc[-1]
#         stop_loss = last_close * 0.95

#         # sideways = is_sideways_market(hist)

#         # Calculate average DlyQttoTradedQty
#         if len(hist2) >= 30:
#             hist['%DlyQttoTradedQty'] = pd.to_numeric(hist2['%DlyQttoTradedQty'], errors='coerce')
#             avg_10_days = hist['%DlyQttoTradedQty'].tail(30).mean()
#         else:
#             avg_10_days = None

#         if avg_10_days is not None:
#             hist['Signal'] = hist.apply(generate_signals, axis=1, args=(avg_10_days,))

#         # Get the last 30 days of data including relevant calculations
#         last_30_days = hist.tail(30)

#         # Replace NaN and infinite values with None for JSON serialization
#         last_30_days_dict = last_30_days.replace([np.nan, np.inf, -np.inf], None).to_dict(orient="records")

#         # Prepare the final analysis results
#         analysis_results = {
#             "support": support,
#             "resistance": resistance,
#             "stopLoss": stop_loss,
#             "10_day_avg_DlyQttoTradedQty": avg_10_days,
#             # "sideways": sideways,
#             "VWAP": last_30_days['VWAP'].iloc[-1],  # VWAP for the most recent day
#             "data": last_30_days_dict
#         }

#         return analysis_results

#     except ValueError as ve:
#         return JSONResponse(content={"error": f"ValueError: {str(ve)}"}, status_code=400)
    
#     except Exception as e:
#         return JSONResponse(content={"error": str(e)}, status_code=500)

import yfinance as yf
import pandas as pd
import numpy as np

def dataAnalysis(sym, period):
    try:
        ySym = sym + ".NS"
        customDate = calculate_date_range(period)

        # Fetch historical stock data from yfinance
        hist = yf.download(ySym, period=period)
        
        # Fetch additional data from capital_market library
        hist2 = capital_market.price_volume_and_deliverable_position_data(sym, 
                from_date=customDate['from_date'], to_date=customDate['to_date'])
        
        # Check if data is available
        if hist.empty or hist2.empty:
            return JSONResponse(content={"error": "No data available. Check the symbol or period."}, status_code=404)

        # Reset index to move Date from index to a column
        hist = hist.reset_index()

        # Flatten MultiIndex columns if present
        if isinstance(hist.columns, pd.MultiIndex):
            hist.columns = ['_'.join(col).strip() for col in hist.columns.values]

        # Find Close price column dynamically (usually 'Close' or 'Close_<symbol>')
        close_col = [col for col in hist.columns if 'Close' in col]
        if not close_col:
            return JSONResponse(content={"error": "Close price data not found."}, status_code=404)
        
        # Extract and clean Close price series
        close_series = hist[close_col[0]].astype(str).str.replace(',', '').str.strip()
        hist['Close'] = pd.to_numeric(close_series, errors='coerce')
        
        # Find Volume column dynamically
        volume_col = [col for col in hist.columns if 'Volume' in col]
        if not volume_col:
            return JSONResponse(content={"error": "Volume data not found."}, status_code=404)
        
        # Extract and clean Volume series
        volume_series = hist[volume_col[0]].astype(str).str.replace(',', '').str.strip()
        hist['Volume'] = pd.to_numeric(volume_series, errors='coerce')

        # Forward fill Close price to fill missing values if any
        hist['Close'].fillna(method='ffill', inplace=True)

        # Validate that Volume and Close have valid data
        if hist['Volume'].isnull().all() or hist['Close'].isnull().all():
            return JSONResponse(content={"error": "No valid data available for analysis."}, status_code=404)

        # VWAP Calculation
        hist['Price_Volume'] = hist['Close'] * hist['Volume']
        cumulative_price_volume = hist['Price_Volume'].cumsum()
        cumulative_volume = hist['Volume'].cumsum()
        hist['VWAP'] = np.where(cumulative_volume != 0, cumulative_price_volume / cumulative_volume, 0)

        # Require at least 200 data points for longest MA (change as needed)
        if len(hist) < 200:
            return JSONResponse(content={"error": "Not enough data points to calculate moving averages."}, status_code=400)

        # Calculate moving averages
        hist['50_MA'] = hist['Close'].rolling(window=50).mean()
        hist['100_MA'] = hist['Close'].rolling(window=100).mean()
        hist['200_MA'] = hist['Close'].rolling(window=200).mean()

        # Shifted previous day MAs for crossover checks
        hist['Prev_50_MA'] = hist['50_MA'].shift(1)
        hist['Prev_200_MA'] = hist['200_MA'].shift(1)

        hist['21_MA'] = hist['Close'].rolling(window=21).mean()
        hist['44_MA'] = hist['Close'].rolling(window=44).mean()
        hist['Prev_21_MA'] = hist['21_MA'].shift(1)
        hist['Prev_44_MA'] = hist['44_MA'].shift(1)

        # Exponential moving averages
        hist['44_EMA'] = hist['Close'].ewm(span=44, adjust=False).mean()
        hist['100_EMA'] = hist['Close'].ewm(span=100, adjust=False).mean()

        # Volume moving averages if enough data
        if len(hist) >= 50:
            hist['10_MA_Volume'] = hist['Volume'].rolling(window=10).mean()
            hist['20_MA_Volume'] = hist['Volume'].rolling(window=20).mean()
            hist['50_MA_Volume'] = hist['Volume'].rolling(window=50).mean()
        else:
            hist['10_MA_Volume'] = None
            hist['20_MA_Volume'] = None
            hist['50_MA_Volume'] = None

        # Support and Resistance
        support, resistance = calculate_support_resistance(hist)

        # Indicators: RSI, MACD, Bollinger Bands
        hist['RSI'] = calculate_rsi(hist)
        hist['MACD'], hist['MACD_Signal'] = calculate_macd(hist)
        hist['Upper_Band'], hist['Lower_Band'] = calculate_bollinger_bands(hist)

        # Rate of Change (ROC)
        hist['ROC'] = ((hist['Close'].iloc[-1] - hist['Close'].iloc[0]) / hist['Close'].iloc[0]) * 100

        # Stop-loss (5% below last close)
        last_close = hist['Close'].iloc[-1]
        stop_loss = last_close * 0.95

        # Use hist2 for average %DlyQttoTradedQty if available
        if len(hist2) >= 30:
            hist['%DlyQttoTradedQty'] = pd.to_numeric(hist2['%DlyQttoTradedQty'], errors='coerce')
            avg_10_days = hist['%DlyQttoTradedQty'].tail(30).mean()
        else:
            avg_10_days = None

        # Generate signals if possible
        if avg_10_days is not None:
            hist['Signal'] = hist.apply(generate_signals, axis=1, args=(avg_10_days,))
        else:
            hist['Signal'] = None

        # Prepare last 30 days data for response
        last_30_days = hist.tail(30)
        last_30_days_dict = last_30_days.replace([np.nan, np.inf, -np.inf], None).to_dict(orient="records")
        last_30_days_dict = pd.DataFrame(last_30_days_dict).sort_values(by='Date_', ascending=False).to_dict(orient="records")

        # Final result dictionary
        analysis_results = {
            "support": support,
            "resistance": resistance,
            "stopLoss": stop_loss,
            "10_day_avg_DlyQttoTradedQty": avg_10_days,
            "VWAP": last_30_days['VWAP'].iloc[-1],
            "ROC": hist['ROC'].iloc[-1],
            "data": last_30_days_dict,
            "latest_signal": hist['Signal'].iloc[-1] if hist['Signal'] is not None else None
        }

        return analysis_results

    except Exception as e:
        tb = traceback.format_exc()
        return JSONResponse(content={
            "errorException": str(e),
            "traceback": tb
        }, status_code=500)

def replace_invalid_floats(obj):
    if isinstance(obj, dict):
        return {k: replace_invalid_floats(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [replace_invalid_floats(i) for i in obj]
    elif isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj
    else:
        return obj
        
@app.get("/swing-trade/{sym}/{period}")
def get_swing_trade(sym: str, period: str):
    try:
        analysis_results = dataAnalysis(sym, period)
        
        # Clean nan/inf from results
        safe_results = replace_invalid_floats(analysis_results)
        
        return JSONResponse(content=jsonable_encoder(safe_results), status_code=200)
    
    except Exception as e:
        tb = traceback.format_exc()
        return JSONResponse(content={
            "errorException": str(e),
            "traceback": tb
        }, status_code=500)
    
def dataAnalysisBasedOnDelivery(stock_symbols, period, threshold):
    try:
        results = []
        # customDate = calculate_date_range(period)  # Calculate the date range for the query
        
        # Function to process data for each stock
        def process_stock(sym):
            # Construct the cache key
            cache_key = f'stock_data:{sym}'
           # Get the data from Redis
            hist = redis_client.get(cache_key)

            if hist:
                # Decode bytes if necessary
                if isinstance(hist, bytes):
                    hist = hist.decode('utf-8')

                # Check if hist is already a list
                if isinstance(hist, str):
                    hist = json.loads(hist)  # Convert from JSON string to Python object
                elif isinstance(hist, list):
                    # If hist is already a list, no need to load JSON
                    pass
                else:
                    logging.warning(f"Unexpected data type for {sym}: {type(hist)}")
                    return None
            
            else:
                logging.warning(f"No data found for symbol: {sym}")
                return None

            # Ensure hist is a DataFrame
            hist = pd.DataFrame(hist)

            logging.info(f"Request received for {sym}: {hist}") 

            # Ensure the column exists before performing operations
            if '%DlyQttoTradedQty' in hist.columns:
                hist['%DlyQttoTradedQty'] = pd.to_numeric(hist['%DlyQttoTradedQty'], errors='coerce')
            
            
            # Check if we have at least 30 days of data
            if len(hist) >= 30:
                avg_30_days = hist['%DlyQttoTradedQty'].tail(30).mean()
            else:
                avg_30_days = None

            logging.info(avg_30_days)

            if pd.notnull(avg_30_days) and avg_30_days > threshold:
                return {
                    "symbol": sym,
                    'Avg_Delivery_Percentage': avg_30_days
                }
            return None

        # Use concurrent futures to process stocks in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(process_stock, symbol) for symbol in stock_symbols]
        
            # Collect results as futures complete
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result:
                    results.append(result)

        # Ensure the function returns an empty DataFrame if no results are found
        if not results:
            return pd.DataFrame()  # Return an empty DataFrame if no stocks match the criteria

        return pd.DataFrame(results)  # Return DataFrame of results

    except Exception as e:
        print(f"Error during stock analysis: {e}")
        return None

# API endpoint to trigger the stock scanner
@app.get('/stocks-scanner/{period}/{threshold}')
def stock_scanner(period: str, threshold: int):
    try: 
        nifty_data = capital_market.equity_list()  # Fetch list of stocks
        if isinstance(nifty_data, pd.DataFrame):
            nifty_data_clean = nifty_data.replace([np.nan, np.inf, -np.inf], None)

        if ' SERIES' in nifty_data_clean.columns:  # Assuming the key column is 'Index Category'
            stocks = nifty_data[nifty_data[' SERIES'] == 'EQ']['SYMBOL']
        
        # Analyze historical data for each stock based on the delivery percentage
        analysis = dataAnalysisBasedOnDelivery(stocks, period, threshold)

        # Handle case when analysis returns an empty DataFrame or None
        if analysis is None or analysis.empty:
            return JSONResponse(content={"error": "No stocks found based on the criteria"}, status_code=200)
    
        # Convert the DataFrame to a list of dictionaries and return it
        return JSONResponse(content={"stocks": analysis.to_dict(orient='records')})    
    except ValueError as ve:
        return JSONResponse(content={"error": f"ValueError: {str(ve)}"}, status_code=400)

def fetch_stock_data(stock_symbols, period):
    """
    Fetch historical stock data for given symbols and period.
    
    Parameters:
        stock_symbols (list): List of stock symbols to fetch data for.
        period (str): The period to fetch data for.

    Returns:
        pd.DataFrame: DataFrame containing the historical stock data.
    """
    try:
        custom_date = calculate_date_range(period)  # Calculate the date range for the query

        # Function to process data for each stock
        def process_stock(sym):
            hist = capital_market.price_volume_and_deliverable_position_data(
                sym, from_date=custom_date['from_date'], to_date=custom_date['to_date']
            )

            return {"symbol": sym, "data": hist}

        # Use concurrent futures to process stocks in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(process_stock, symbol): symbol for symbol in stock_symbols}
        
            # Collect results as futures complete
            results = []  # Initialize results list for returning
            for future in concurrent.futures.as_completed(futures):
                symbol = futures[future]  # Get the symbol associated with this future
                try:
                    result = future.result()
                    if result and 'data' in result:
                        results.append(result)
                        cache_key = f'stock_data:{symbol}'  # Construct cache key
                        redis_client.set(cache_key, result['data'].to_json(orient='records'))  # Store data in Redis
                        logging.info(f"Stored data for {symbol} in Redis under key: {cache_key}")
                except Exception as e:
                    logging.error(f"Error fetching data for {symbol}: {e}")

        # Ensure the function returns an empty DataFrame if no results are found
        if not results:
            return pd.DataFrame()  # Return an empty DataFrame if no stocks match the criteria

        # Combine results into a DataFrame
        return pd.DataFrame(results)

    except Exception as e:
        logging.error(f"Error during stock analysis: {e}")
        return pd.DataFrame()  # Return empty DataFrame on error


# def combinedPriceAndDeliveryData(sym, period):
#     try:
#         custom_date = calculate_date_range(period)
#         logging.error(f"{sym}: {custom_date}: {period}")
#         # Fetch price volume data
#         priceVolumeData = capital_market.price_volume_data(
#             symbol=sym, from_date=custom_date['from_date'], to_date=custom_date['to_date']
#         )

#         # Fetch delivery data
#         deliveryData = capital_market.deliverable_position_data(
#             symbol=sym, from_date=custom_date['from_date'], to_date=custom_date['to_date']
#         )

#         # Ensure both DataFrames have a 'Date' column for merging
#         if 'Date' in priceVolumeData.columns and 'Date' in deliveryData.columns:
#             # Attempt to merge the data
#             combinedData = pd.merge(priceVolumeData, deliveryData, on='Date', how='inner')
#         else:
#             print("The datasets do not have a common 'Date' column.")
#             combinedData = None

#         # Return the combined data
#         if combinedData is not None:
#             return combinedData
#         else:
#             print("Data could not be merged.")
#             return None

#     except UnicodeDecodeError as e:
#         logging.error(f"Encoding error while processing stock {sym}: {e}")
#         return None
#     except Exception as e:
#         logging.error(f"Error combining stock {sym}: {e}")
#         return None


# def fetch_stock_data(stock_symbols, period):
#     try:        
#         # Function to process data for each stock
#         def process_stock(sym):
#             try:
#                 hist = combinedPriceAndDeliveryData(sym, period)
                
#                 # Check if hist is a valid DataFrame
#                 if isinstance(hist, pd.DataFrame):
#                     return {"symbol": sym, "data": hist}
#                 else:
#                     logging.error(f"Invalid data for {sym}: {hist}")
#                     return None

#             except Exception as e:
#                 logging.error(f"Error processing stock {sym}: {e}")
#                 return None

#         # Use concurrent futures to process stocks in parallel
#         with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
#             futures = {executor.submit(process_stock, symbol): symbol for symbol in stock_symbols}
        
#             # Collect results as futures complete
#             results = []  # Initialize results list for returning
#             for future in concurrent.futures.as_completed(futures):
#                 symbol = futures[future]  # Get the symbol associated with this future
#                 try:
#                     result = future.result()  # Fetch the result from the future
                    
#                     if result and 'data' in result:
#                         data = result['data']
#                         if isinstance(data, pd.DataFrame) and not data.empty:
#                             # Store the data in Redis
#                             cache_key = f'stock_data:{symbol}'  # Construct cache key
#                             redis_client.set(cache_key, data.to_json(orient='records'))
#                             logging.info(f"Stored data for {symbol} in Redis under key: {cache_key}")
#                         else:
#                             logging.error(f"Invalid or empty data for {symbol}: {data}")
#                     else:
#                         logging.error(f"No data found for {symbol}")
#                 except Exception as e:
#                     logging.error(f"Error retrieving result for {symbol}: {e}")
            
#         return pd.DataFrame(results) if results else pd.DataFrame()

#     except Exception as e:
#         logging.error(f"Error during stock analysis: {e}")
#         return pd.DataFrame()  # Return empty DataFrame on error


@app.get('/store-stock-data/{period}')
def store_stock_data(period):
    """
    Endpoint to store stock data in Redis for a specified period.
    
    Parameters:
        period (str): The period to fetch stock data for.
    
    Returns:
        dict: Response indicating success or error.
    """
    try: 
        nifty_data = capital_market.equity_list()  # Fetch list of stocks
        
        if isinstance(nifty_data, pd.DataFrame):
            nifty_data_clean = nifty_data.replace([np.nan, np.inf, -np.inf], None)

            if ' SERIES' in nifty_data_clean.columns:
                stocks = nifty_data_clean[nifty_data_clean[' SERIES'] == 'EQ']['SYMBOL'].tolist()
                
            # Call fetch_stock_data with the list of stock symbols
            fetch_stock_data(stocks, period)

        return {"message": "Stock data stored successfully."}

    except Exception as e:
        logging.error(f"Error in storing stock data: {e}")
        return {"error": str(e)}


@app.get("/fii-dii-activity")
def get_fii_dii_activity():
    try:
        # Fetch data from NSE using the price_volume_and_deliverable_posymsition_data function
        nifty_data = capital_market.fii_dii_trading_activity()
        
        if isinstance(nifty_data, pd.DataFrame):
            # Replace NaN and infinite values with None
            nifty_data_clean = nifty_data.replace([np.nan, np.inf, -np.inf], None)
            
            # Convert DataFrame to dictionary
            nifty_data_dict = nifty_data_clean.to_dict(orient="records")
            
            # Use jsonable_encoder to ensure it's JSON-serializable
            return jsonable_encoder(nifty_data_dict)
        else:
            return {"error": "Unexpected data format"}
    except Exception as e:
        return {"error": str(e)}
    
@app.get("/etfs")
def get_etf_list():
    try:
        # Load the CSV file
        etf_data = pd.read_csv('./etfs.csv')

        # Rename the relevant column (assuming 'Unnamed: 0' holds the ETF codes)
        etf_data_cleaned = etf_data.rename(columns={'Unnamed: 0': 'ETF_Code'})

        # Skip the first row if it contains headers or irrelevant data
        etf_data_cleaned = etf_data_cleaned.iloc[1:]

        # Clean the ETF codes: Remove 'NSE:' prefix and handle rows without 'NSE:'
        etf_data_cleaned['Formatted_Ticker'] = etf_data_cleaned['ETF_Code'].apply(
            lambda x: x.split(':')[1] + ".NS" if pd.notna(x) and "NSE:" in x else None
        )

        # Filter out any None (invalid) tickers
        etf_codes = etf_data_cleaned['Formatted_Ticker'].dropna().tolist()

        return {"etf_tickers": etf_codes}

    except Exception as e:
        return {"error": str(e)}
    
@app.get("/etf/{ticker}")
def get_etf_info(ticker: str):
    try:
        # Get data from yfinance
        etf = yf.Ticker(ticker)

        # Fetch basic information
        etf_info = etf.info

        return {"ticker": ticker, "info": etf_info, "iNav": calculate_inav(ticker)}

    except Exception as e:
        return {"error": str(e)}


def calculate_inav(etf_ticker):
    # Fetch ETF info using yfinance
    etf = yf.Ticker(etf_ticker)
    # data = etf.funds_data
    
    # Get the holdings (underlying assets) of the ETF
    try:
        holdings = etf.equity_holdings
        logging.info(holdings)
    except Exception as e:
        print(f"Error fetching holdings: {str(e)}")
        return None

    # If holdings are not available, return None
    if holdings is None or holdings.empty:
        print("No holdings data available for the ETF.")
        return None

    # Extract relevant columns: 'Symbol', 'Holding', 'Weight'
    # Adjust the columns as per the actual data structure
    holdings = holdings[['Symbol', 'Holding', 'Weight']]
    
    # Initialize a list to store the values of each holding
    total_value = 0.0

    # Loop through each holding to calculate its value
    for index, row in holdings.iterrows():
        symbol = row['Symbol']
        weight = row['Weight']
        
        # Get the current price of the underlying asset
        asset = yf.Ticker(symbol)
        current_price = asset.history(period="1d")['Close'].iloc[-1]
        
        # Calculate the value of the holding
        holding_value = current_price * row['Holding'] * weight
        
        # Add to total value
        total_value += holding_value

    # Return the calculated iNAV
    return total_value

@app.delete('/delete-history')
def history_delete():
    try: 
        nifty_data = capital_market.equity_list()  # Fetch list of stocks
            
        if isinstance(nifty_data, pd.DataFrame):
            nifty_data_clean = nifty_data.replace([np.nan, np.inf, -np.inf], None)
            
            if ' SERIES' in nifty_data_clean.columns:
                stocks = nifty_data_clean[nifty_data_clean[' SERIES'] == 'EQ']['SYMBOL'].tolist()
                BEStocks = nifty_data_clean[nifty_data_clean[' SERIES'] == 'BE']['SYMBOL'].tolist()

                combineStocks = stocks + BEStocks
                
            def delete_stock_data(symbol):
                """Delete stock data for a given symbol from Redis."""
                cache_key = f'stock_data:{symbol}'  # Construct cache key
                result = redis_client.delete(cache_key)  # Delete the data from Redis
                return {
                    "symbol": symbol,
                    "deleted": result
                }

            # Use concurrent futures to process stocks in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                futures = {executor.submit(delete_stock_data, symbol): symbol for symbol in combineStocks}
            
                # Collect results as futures complete
                results = []  # Initialize results list for returning
                for future in concurrent.futures.as_completed(futures):
                    symbol = futures[future]  # Get the symbol associated with this future
                    try:
                        result = future.result()
                        results.append(result)
                        logging.info(f"Deleted data for {symbol} from Redis.")
                    except Exception as e:
                        logging.error(f"Error deleting data for {symbol}: {e}")

            return {"message": "Stock data deleted successfully."}

    except Exception as e:
        logging.error(f"Error in storing stock data: {e}")
        return {"error": str(e)}

                


# Run the app with: uvicorn main:app --reload
