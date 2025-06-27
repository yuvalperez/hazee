#!/usr/bin/env python3
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow debug messages

import ccxt
import numpy as np
import pandas as pd
import ta
import joblib
import random
import threading
import time
from datetime import datetime, timedelta
import logging
from logging.handlers import RotatingFileHandler
import json
import boto3
from botocore.exceptions import NoCredentialsError
import requests
import sys

# ========== Configuration ==========
DEFAULT_SYMBOLS = ['BTC/USDT', 'ETH/USDT']
DEFAULT_TIMEFRAME = '1h'
INITIAL_BALANCE = 10000
RISK_PER_TRADE = 0.01
FEATURES = ['returns', 'atr', 'rsi', 'macd', 'ema_20', 'ema_50', 'adx', 'obv']

class AdvancedAITrader:
    def __init__(self, headless=True):
        # Initialize logging first
        self.setup_logging()
        
        # Core parameters
        self.learning_rate = 0.001
        self.exploration_rate = 0.3
        self.min_exploration = 0.05
        self.exploration_decay = 0.995
        self.sequence_length = 24
        self.headless = headless
        self.running = False

        # Exchange connection
        self.exchange = self.connect_to_exchange()
        
        # Initialize models and data
        self.models = {}
        self.scalers = {}
        self.historical_data = {}
        self.trade_history = []
        
        # Risk management
        self.max_daily_risk = 0.05
        self.current_daily_risk = 0
        
        # Portfolio tracking
        self.portfolio = {symbol: {'balance': INITIAL_BALANCE/len(DEFAULT_SYMBOLS), 
                         'positions': 0} for symbol in DEFAULT_SYMBOLS}

    def setup_logging(self):
        """Configure professional logging system"""
        self.logger = logging.getLogger('TradingBot')
        self.logger.setLevel(logging.INFO)
        
        # Create log formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)
        
        # File handler with rotation
        fh = RotatingFileHandler('trading_bot.log', maxBytes=5*1024*1024, backupCount=3)
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
        
        self.log("Logging system initialized", level='INFO')

    def log(self, message, level='INFO'):
        """Unified logging method"""
        if level == 'INFO':
            self.logger.info(message)
        elif level == 'WARNING':
            self.logger.warning(message)
        elif level == 'ERROR':
            self.logger.error(message)
            self.send_alert(f"ERROR: {message}")
        elif level == 'DEBUG':
            self.logger.debug(message)

    def connect_to_exchange(self):
        """Initialize exchange connection with error handling"""
        try:
            exchange = ccxt.bybit({
                'options': {'defaultType': 'future'},
                'enableRateLimit': True,
                'timeout': 30000
            })
            
            # Load API keys if available
            if os.path.exists('api_keys.json'):
                with open('api_keys.json') as f:
                    keys = json.load(f)
                exchange.apiKey = keys['apiKey']
                exchange.secret = keys['secret']
            
            exchange.load_markets()
            self.log(f"Connected to {exchange.name}", level='INFO')
            return exchange
        except Exception as e:
            self.log(f"Connection failed: {e}", level='ERROR')
            return None

    def fetch_historical_data(self, symbol, limit=1000):
        """Fetch OHLCV data with retry logic"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                ohlcv = self.exchange.fetch_ohlcv(symbol, self.timeframe, limit=limit)
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                return df
            except Exception as e:
                self.log(f"Attempt {attempt + 1} failed for {symbol}: {e}", level='WARNING')
                if attempt == max_retries - 1:
                    return None
                time.sleep(5)

    def calculate_features(self, df):
        """Calculate technical indicators"""
        try:
            # Price features
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log(df['close']/df['close'].shift(1))
            
            # Volatility
            df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
            df['volatility'] = df['close'].rolling(20).std() / df['close'].rolling(20).mean()
            
            # Momentum
            df['rsi'] = ta.momentum.rsi(df['close'], window=14)
            df['macd'] = ta.trend.macd_diff(df['close'])
            
            # Trend
            df['ema_20'] = ta.trend.ema_indicator(df['close'], window=20)
            df['ema_50'] = ta.trend.ema_indicator(df['close'], window=50)
            df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'], window=14)
            
            # Volume
            df['obv'] = ta.volume.on_balance_volume(df['close'], df['volume'])
            
            # Target variable
            df['future_return'] = df['close'].pct_change().shift(-1)
            df['action_target'] = np.where(df['future_return'] > 0.002, 1, 
                                         np.where(df['future_return'] < -0.002, 2, 0))
            
            df.dropna(inplace=True)
            return df
        except Exception as e:
            self.log(f"Feature calculation error: {e}", level='ERROR')
            return None

    def run(self):
        """Main trading loop"""
        self.log("Starting trading bot", level='INFO')
        self.running = True
        
        try:
            while self.running:
                for symbol in DEFAULT_SYMBOLS:
                    try:
                        # Fetch and process data
                        df = self.fetch_historical_data(symbol)
                        if df is None:
                            continue
                            
                        df = self.calculate_features(df)
                        if df is None or len(df) < self.sequence_length:
                            continue
                            
                        # Make trading decision
                        action, _ = self.analyze_market(df)
                        if action != 0:  # 0 = no action
                            self.execute_trade(symbol, action, df.iloc[-1])
                            
                    except Exception as e:
                        self.log(f"Error processing {symbol}: {e}", level='ERROR')
                
                # Sleep until next candle
                time.sleep(self.get_sleep_time())
                
        except KeyboardInterrupt:
            self.log("Bot stopped by user", level='INFO')
        except Exception as e:
            self.log(f"Fatal error in main loop: {e}", level='ERROR')
        finally:
            self.running = False
            self.save_state()

    def execute_trade(self, symbol, action, last_row):
        """Execute trade with risk management"""
        try:
            balance = self.portfolio[symbol]['balance']
            position_size = self.calculate_position_size(last_row['atr'], balance)
            
            if position_size <= 0:
                return
                
            if action == 1:  # Buy
                order = self.exchange.create_market_buy_order(
                    symbol=symbol,
                    amount=position_size
                )
                self.portfolio[symbol]['positions'] += position_size
            elif action == 2:  # Sell
                order = self.exchange.create_market_sell_order(
                    symbol=symbol,
                    amount=position_size
                )
                self.portfolio[symbol]['positions'] -= position_size
                
            self.log(f"Executed {symbol} trade: {order}", level='INFO')
            
        except Exception as e:
            self.log(f"Trade execution failed: {e}", level='ERROR')

    def save_state(self):
        """Save bot state to disk"""
        try:
            state = {
                'models': self.models,
                'scalers': self.scalers,
                'trade_history': self.trade_history,
                'portfolio': self.portfolio
            }
            joblib.dump(state, 'bot_state.pkl')
            self.log("Bot state saved", level='INFO')
        except Exception as e:
            self.log(f"Failed to save state: {e}", level='ERROR')

    def load_state(self):
        """Load bot state from disk"""
        try:
            if os.path.exists('bot_state.pkl'):
                state = joblib.load('bot_state.pkl')
                self.models = state.get('models', {})
                self.scalers = state.get('scalers', {})
                self.trade_history = state.get('trade_history', [])
                self.portfolio = state.get('portfolio', {})
                self.log("Bot state loaded", level='INFO')
                return True
            return False
        except Exception as e:
            self.log(f"Failed to load state: {e}", level='ERROR')
            return False

    def send_alert(self, message):
        """Send notification (placeholder for actual alert system)"""
        if not hasattr(self, 'last_alert_time'):
            self.last_alert_time = datetime.now() - timedelta(minutes=5)
            
        if (datetime.now() - self.last_alert_time).total_seconds() > 300:  # 5 min cooldown
            print(f"ALERT: {message}")
            self.last_alert_time = datetime.now()

if __name__ == "__main__":
    # Initialize and run the bot
    trader = AdvancedAITrader(headless=True)
    
    # Load previous state if available
    trader.load_state()
    
    # Start trading
    try:
        trader.run()
    except KeyboardInterrupt:
        trader.log("Shutting down gracefully...", level='INFO')
    finally:
        trader.save_state()
