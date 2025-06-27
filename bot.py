import ccxt
import numpy as np
import pandas as pd
import ta
import joblib
import random
import threading
import time
from datetime import datetime, timedelta
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.gridspec import GridSpec
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tqdm import tqdm
from scipy.stats import norm
import seaborn as sns
import json
import os
import boto3
from botocore.exceptions import NoCredentialsError
import logging
from logging.handlers import RotatingFileHandler
import requests
import sys

# ========== Configuration ==========
DEFAULT_SYMBOLS = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'ADA/USDT', 'DOT/USDT']
DEFAULT_TIMEFRAME = '1h'
INITIAL_BALANCE = 10000
RISK_PER_TRADE = 0.01  # 1% of balance per trade
FEATURES = ['returns', 'atr', 'rsi', 'macd', 'ema_20', 'ema_50', 
           'adx', 'obv', 'cmf', 'volatility', 'log_returns', 'stoch_rsi']
AWS_S3_BUCKET = 'your-s3-bucket-name'
TELEGRAM_BOT_TOKEN = 'your-telegram-bot-token'
TELEGRAM_CHAT_ID = 'your-chat-id'

class AdvancedAITrader:
    def __init__(self, master=None):
        # Core parameters
        self.learning_rate = 0.001
        self.exploration_rate = 0.3
        self.min_exploration = 0.05
        self.exploration_decay = 0.995
        self.sequence_length = 24  # 24 hours window
        self.position_sizing_enabled = True
        self.walk_forward_windows = 5
        self.visualization_enabled = True
        self.monte_carlo_simulations = 1000
        self.live_trading_enabled = False
        self.running = False
        self.symbols = DEFAULT_SYMBOLS.copy()
        self.timeframe = DEFAULT_TIMEFRAME
        self.current_symbol = self.symbols[0]
        self.cloud_mode = False
        self.telegram_alerts = False
        
        # Core components
        self.exchange = self.connect_to_exchange()
        self.scalers = {}
        self.models = {}  # One model per symbol
        self.trade_history = []
        self.performance_metrics = {
            'win_rate': [],
            'profit_factor': [],
            'max_drawdown': [],
            'sharpe_ratio': []
        }
        self.is_scalers_fitted = False
        self.is_model_trained = False
        self.historical_data = {}
        self.portfolio = {symbol: {'balance': INITIAL_BALANCE/len(self.symbols), 
                         'positions': 0} for symbol in self.symbols}
        
        # Risk management
        self.max_daily_risk = 0.05  # 5% max daily loss
        self.current_daily_risk = 0
        
        # Notification system
        self.notifications = []
        self.last_notification_time = datetime.now()
        
        # AWS components
        self.s3_client = boto3.client('s3') if self.cloud_mode else None
        
        # Initialize logging
        self.setup_logging()
        
        # Initialize GUI if master is provided
        if master:
            self.setup_gui(master)
            
        # Load configuration
        self.load_config()

    # ========== Core Methods ==========
    
    def connect_to_exchange(self):
        """Initialize exchange connection with error handling"""
        try:
            exchange = ccxt.bybit({
                'options': {'defaultType': 'future' if self.live_trading_enabled else 'spot'},
                'enableRateLimit': True,
                'timeout': 30000
            })
            
            if self.live_trading_enabled:
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
    
    def fetch_historical_data(self, symbol=None, limit=2000):
        """Fetch OHLCV data with retry logic"""
        symbol = symbol or self.current_symbol
        max_retries = 3
        for attempt in range(max_retries):
            try:
                ohlcv = self.exchange.fetch_ohlcv(symbol, self.timeframe, limit=limit)
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                self.historical_data[symbol] = df
                return df
            except Exception as e:
                self.log(f"Attempt {attempt + 1} failed for {symbol}: {e}", level='WARNING')
                if attempt == max_retries - 1:
                    return None
                time.sleep(5)
    
    # ... [Previous methods remain the same, with added error handling and logging] ...

    def setup_logging(self):
        """Configure professional logging system"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                RotatingFileHandler('trading_bot.log', maxBytes=5*1024*1024, backupCount=3),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
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
        
        # Add to GUI console if available
        if hasattr(self, 'console'):
            self.console.insert(tk.END, f"[{datetime.now()}] {message}\n")
            self.console.see(tk.END)
    
    def send_alert(self, message):
        """Send notification via Telegram"""
        if not self.telegram_alerts:
            return
            
        if (datetime.now() - self.last_notification_time).total_seconds() < 60:
            return  # Rate limiting
            
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
            params = {
                'chat_id': TELEGRAM_CHAT_ID,
                'text': message
            }
            requests.post(url, params=params)
            self.last_notification_time = datetime.now()
        except Exception as e:
            self.log(f"Failed to send Telegram alert: {e}", level='ERROR')

    # ... [GUI setup methods would go here] ...

    def start_live_trading(self):
        """Start the live trading thread"""
        if not self.running:
            self.running = True
            self.trading_thread = threading.Thread(target=self.run_live, daemon=True)
            self.trading_thread.start()
            self.log("Live trading started", level='INFO')
            return True
        return False
    
    def stop_live_trading(self):
        """Stop the live trading thread"""
        if self.running:
            self.running = False
            if hasattr(self, 'trading_thread'):
                self.trading_thread.join(timeout=5)
            self.log("Live trading stopped", level='INFO')
            return True
        return False
    
    def run_live(self):
        """Main live trading loop"""
        while self.running:
            try:
                # Update market data
                for symbol in self.symbols:
                    df = self.fetch_historical_data(symbol)
                    if df is not None:
                        df = self.calculate_features(df)
                        if len(df) > self.sequence_length:
                            action, _ = self.analyze_market(df)
                            self.execute_live_trade(symbol, action, df.iloc[-1])
                
                # Sleep until next candle
                now = datetime.now()
                next_update = now + timedelta(minutes=self.get_timeframe_minutes())
                sleep_time = (next_update - now).total_seconds()
                time.sleep(max(0, sleep_time))
                
            except Exception as e:
                self.log(f"Live trading error: {e}", level='ERROR')
                time.sleep(60)
    
    def execute_live_trade(self, symbol, action, last_row):
        """Execute live trade on exchange"""
        if action == 0:  # No action
            return
            
        try:
            balance = self.portfolio[symbol]['balance']
            position_size = self.calculate_position_size(last_row['atr'], balance)
            
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
            self.send_alert(f"New trade executed: {symbol} {action} at {last_row['close']}")
            
        except Exception as e:
            self.log(f"Trade execution failed: {e}", level='ERROR')
    
    # ... [Additional methods for cloud integration, GUI updates, etc.] ...

    def save_config(self):
        """Save current configuration to file"""
        config = {
            'symbols': self.symbols,
            'timeframe': self.timeframe,
            'risk_per_trade': RISK_PER_TRADE,
            'max_daily_risk': self.max_daily_risk,
            'live_trading': self.live_trading_enabled,
            'telegram_alerts': self.telegram_alerts,
            'cloud_mode': self.cloud_mode
        }
        
        try:
            with open('config.json', 'w') as f:
                json.dump(config, f)
            if self.cloud_mode:
                self.s3_client.upload_file('config.json', AWS_S3_BUCKET, 'config.json')
        except Exception as e:
            self.log(f"Failed to save config: {e}", level='ERROR')
    
    def load_config(self):
        """Load configuration from file"""
        try:
            if self.cloud_mode:
                self.s3_client.download_file(AWS_S3_BUCKET, 'config.json', 'config.json')
                
            with open('config.json') as f:
                config = json.load(f)
                self.symbols = config.get('symbols', DEFAULT_SYMBOLS)
                self.timeframe = config.get('timeframe', DEFAULT_TIMEFRAME)
                global RISK_PER_TRADE
                RISK_PER_TRADE = config.get('risk_per_trade', 0.01)
                self.max_daily_risk = config.get('max_daily_risk', 0.05)
                self.live_trading_enabled = config.get('live_trading', False)
                self.telegram_alerts = config.get('telegram_alerts', False)
                self.cloud_mode = config.get('cloud_mode', False)
        except FileNotFoundError:
            self.log("No config file found, using defaults", level='INFO')
        except Exception as e:
            self.log(f"Failed to load config: {e}", level='ERROR')

# ========== GUI Application ==========
class TradingApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Advanced AI Trading Bot")
        self.geometry("1200x800")
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        
        self.trader = AdvancedAITrader(self)
        self.create_widgets()
        
    def create_widgets(self):
        # Control Panel
        control_frame = ttk.LabelFrame(self, text="Control Panel", padding=10)
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # ... [GUI widget creation code] ...
        
    def on_close(self):
        if messagebox.askokcancel("Quit", "Do you want to stop trading and quit?"):
            self.trader.stop_live_trading()
            self.trader.save_config()
            self.destroy()

# ========== Main Execution ==========
if __name__ == "__main__":
    app = TradingApp()
    app.mainloop()
