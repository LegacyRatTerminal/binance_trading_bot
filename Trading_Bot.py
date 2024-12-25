import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import os
from logging.handlers import RotatingFileHandler
import ta
import random
import tabulate
import threading
import time
import telegram
from telegram.ext import Updater, CommandHandler
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


class AdvancedSignalGenerator:
    def __init__(self, exchange):
        self.exchange = exchange
        self.model = None
        self.scaler = None
        
    def prepare_features(self, df):
        """
        Generate comprehensive features for machine learning
        """
        # Technical Indicators
        macd = ta.trend.MACD(close=df['close'])
        bollinger = ta.volatility.BollingerBands(close=df['close'])
        
        # Add features to DataFrame
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['bb_upper'] = bollinger.bollinger_hband()
        df['bb_lower'] = bollinger.bollinger_lband()
        
        # Additional features
        df['ma_20'] = df['close'].rolling(window=20).mean()
        df['ma_50'] = df['close'].rolling(window=50).mean()
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        
        # Directional movement
        df['price_change_pct'] = df['close'].pct_change()
        
        return df.dropna()
    
    def train_model(self, symbol, timeframe='15m', lookback=1000, evaluate_performance=True):
        """
        Train a machine learning model for signal generation with optional performance evaluation
        """
        try:
            # Fetch historical data
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=lookback)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # Prepare features
            df = self.prepare_features(df)
            
            # Create target variable (next period price movement)
            df['target'] = np.where(df['price_change_pct'].shift(-1) > 0, 1, 0)
            
            # Select features
            features = [
                'macd', 'macd_signal', 
                'bb_upper', 'bb_lower', 
                'ma_20', 'ma_50', 
                'volume_ma'
            ]
            
            # Prepare data
            X = df[features].dropna()
            y = df['target'].dropna()
            
            # Split data if performance evaluation is requested
            if evaluate_performance:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            else:
                X_train, y_train = X, y
                X_test, y_test = None, None
            
            # Scale features
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            
            # Train Random Forest Classifier
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.model.fit(X_train_scaled, y_train)
            
            # Performance evaluation
            if evaluate_performance and X_test is not None:
                X_test_scaled = self.scaler.transform(X_test)
                y_pred = self.model.predict(X_test_scaled)
                
                print(f"\nModel Performance for {symbol}:")
                print(f"Accuracy: {accuracy_score(y_test, y_pred):.2%}")
                print("\nClassification Report:")
                print(classification_report(y_test, y_pred))
                
                # Feature Importance
                feature_importances = self.model.feature_importances_
                importance_dict = dict(zip(features, feature_importances))
                sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
                
                print("\nFeature Importances:")
                for feature, importance in sorted_importance:
                    print(f"{feature}: {importance:.4f}")
            
            return True
        except Exception as e:
            print(f"Error training model: {e}")
            return False

    def analyze_feature_importance(self):
        """Analyze feature importance of the trained model"""
        if self.model is None:
            print("Model not trained yet")
            return
        
        features = [
            'macd', 'macd_signal', 
            'bb_upper', 'bb_lower', 
            'ma_20', 'ma_50', 
            'volume_ma'
        ]
        
        importances = self.model.feature_importances_
        feature_importance = sorted(zip(features, importances), key=lambda x: x[1], reverse=True)
        
        print("Feature Importances:")
        for feature, importance in feature_importance:
            print(f"{feature}: {importance:.4f}")

    def generate_advanced_signal(self, symbol, timeframe='15m'):
        """
        Generate trading signal using machine learning and technical analysis
        """
        try:
            # Fetch recent data
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=100)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # Prepare features
            df = self.prepare_features(df)
            
            # Select most recent features
            latest_features = df[['macd', 'macd_signal', 
                                  'bb_upper', 'bb_lower', 
                                  'ma_20', 'ma_50', 
                                  'volume_ma']].iloc[-1:]
            
            # Scale features
            if self.scaler is None or self.model is None:
                # Train model if not already trained
                self.train_model(symbol)
            
            latest_features_scaled = self.scaler.transform(latest_features)
            
            # Predict using machine learning model
            ml_prediction = self.model.predict(latest_features_scaled)[0]
            
            # Technical analysis signals
            latest = df.iloc[-1]
            current_price = latest['close']
            
            # Combine ML and Technical Analysis
            long_signal = (
                latest['macd'] > latest['macd_signal'] and
                current_price < latest['bb_lower'] and
                ml_prediction == 1
            )
            
            short_signal = (
                latest['macd'] < latest['macd_signal'] and
                current_price > latest['bb_upper'] and
                ml_prediction == 0
            )
            
            # Determine final signal
            if long_signal:
                return 'long', current_price
            elif short_signal:
                return 'short', current_price
            
            return None, current_price
        
        except Exception as e:
            print(f"Error generating advanced signal: {e}")
            return None, None

class BinanceTradingBot:
    # Color codes as class attributes
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BOLD = '\033[1m'
    END = '\033[0m'

    def display_performance_metrics(self):
        """Display comprehensive performance metrics"""
        print(f"\n{self.BOLD} Performance Metrics{self.END}")
        print(f"{self.BLUE}{'=' * 50}{self.END}")

        # Calculate metrics from trade history
        total_trades = len(self.trade_history)
        if total_trades > 0:
            winning_trades = sum(1 for trade in self.trade_history if trade['profit_loss'] > 0)
            win_rate = (winning_trades / total_trades) * 100
            
            total_profit = sum(trade['profit_loss'] for trade in self.trade_history)
            avg_profit = total_profit / total_trades
            
            # Calculate max drawdown
            cumulative_returns = [trade['profit_loss'] for trade in self.trade_history]
            max_drawdown = min(cumulative_returns) if cumulative_returns else 0

            metrics = [
                ['Total Trades 📈', f"{total_trades}"],
                ['Win Rate 🎯', f"{self.GREEN}{win_rate:.2f}%{self.END}"],
                ['Total Profit 💰', f"{self.GREEN if total_profit > 0 else self.RED}${total_profit:.2f}{self.END}"],
                ['Average Profit/Trade ', f"{self.GREEN if avg_profit > 0 else self.RED}${avg_profit:.2f}{self.END}"],
                ['Max Drawdown 📉', f"{self.RED}${abs(max_drawdown):.2f}{self.END}"],
                ['Active Positions 🔄', f"{self.YELLOW}{len(self.open_trades)}{self.END}"]
            ]

            print(tabulate.tabulate(metrics, tablefmt='fancy_grid'))
        else:
            print(f"{self.YELLOW}No trades executed yet{self.END}")


    def __init__(self, 
                 api_key='API_KEY',
                 secret_key='SECRET_KEY',
                 symbols=['XRP/USDT', 'ADA/USDT', 'LINK/USDT', 'AAVE/USDT', 'AVAX/USDT', 'DOGE/USDT', 'DOT/USDT', 'EGLD/USDT', 'ENJ/USDT', 'FTM/USDT', 'INJ/USDT', 'MANA/USDT', 'RAY/USDT', 'UNI/USDT', 'AGLD/USDT'], 
                 timeframes=['15m', '5m']):
        """
        Enhanced Binance Trading Bot with Trade Execution and Continuous Monitoring
        """
        # Initialize logging FIRST
        self.setup_logging()
        self.trade_monitoring_threads = {}
        
        # Create logs directory if it doesn't exist
        os.makedirs('logs', exist_ok=True)
        
        # Initialize Binance exchange
        try:
            self.exchange = ccxt.binance({
                'apiKey': api_key,
                'secret': secret_key,
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'future'
                }
            })
        except Exception as e:
            self.logger.error(f"Failed to initialize exchange: {e}")
            raise
        
        # Trading Configuration
        self.symbols = symbols
        self.timeframes = timeframes
        self.trade_monitoring_threads = {}
        
        # Trading interval for strategy execution
        self.check_interval = 1  # Check every 1 seconds
        
        # Strategy parameters
        self.macd_crossover_sensitivity = 0.1
        
        # Risk Management Parameters
        self.max_risk_per_trade = 0.40
        self.max_portfolio_risk = 0.40
        self.max_leverage = 20
        self.trade_allocation_percentage = 0.40
        self.take_profit_percentage = 0.01
        self.stop_loss_percentage = 0.01

        # Add these new parameters here
        # Dynamic position management parameters
        self.initial_stop_loss_percentage = 0.01  # 1%
        self.breakeven_profit_level = 0.005  # 0.5%
        self.min_take_profit = 0.01  # 1%
        self.max_take_profit = 0.03  # 3%
        self.trailing_stop_distances = {
            0.01: 0.01,  # 1-2% profit -> 1% trailing distance
            0.02: 0.008, # 2-2.5% profit -> 0.8% trailing distance
            0.025: 0.006 # 2.5-3% profit -> 0.6% trailing distance
        }
        self.max_position_time = 14400  # 4 hours in seconds
        
        # Trade Tracking
        self.open_trades = {}
        self.trade_history = []
        
        # Trading state
        self.is_running = False
        self.trading_thread = None
        
        # Startup message
        startup_msg = ("Enhanced Trading Bot Initialized!")
    
        self.logger.info(startup_msg)

        self.setup_telegram()
        self.send_telegram_message(startup_msg)

    def calculate_performance_metrics(self):
        """Calculate comprehensive trading performance metrics"""
        total_trades = len(self.trade_history)
    
        if total_trades == 0:
            return None
    
        # Calculate win rate
        winning_trades = sum(1 for trade in self.trade_history if trade['profit_loss'] > 0)
        win_rate = (winning_trades / total_trades) * 100
    
        # Total profit/loss
        total_profit = sum(trade['profit_loss'] for trade in self.trade_history)
    
        # Average profit per trade
        avg_profit = total_profit / total_trades
    
        # Maximum drawdown
        cumulative_returns = [trade['profit_loss'] for trade in self.trade_history]
        max_drawdown = min(cumulative_returns) if cumulative_returns else 0
    
        performance_metrics = {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'total_profit': total_profit,
            'average_profit_per_trade': avg_profit,
            'max_drawdown': max_drawdown
        }
    
        return performance_metrics  

    def log_trade_details(self, trade_details):
        """
        Enhanced trade logging with more comprehensive information
        """
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'symbol': trade_details['symbol'],
            'type': trade_details['type'],
            'entry_price': trade_details['entry_price'],
            'exit_price': trade_details.get('exit_price'),
            'profit_loss': trade_details.get('profit_loss', 0),
            'duration': trade_details.get('duration')
        }
    
        # Optional: Write to a CSV or database for long-term analysis
        self.write_trade_log_to_csv(log_entry)  

    def generate_trading_signals(self, symbol, timeframe):
        """Generate trading signals based on MACD and Bollinger Bands"""
        try:
            # Fetch OHLCV data
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=100)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # Calculate indicators
            macd = ta.trend.MACD(close=df['close'])
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            bollinger = ta.volatility.BollingerBands(close=df['close'])
            df['bb_upper'] = bollinger.bollinger_hband()
            df['bb_lower'] = bollinger.bollinger_lband()
            
            latest = df.iloc[-1]
            current_price = latest['close']
            
            # Long signal conditions
            long_signal = (
                
                latest['macd'] > latest['macd_signal'] and
                current_price < latest['bb_lower']
            )
            
            # Short signal conditions
            short_signal = (
            
                latest['macd'] < latest['macd_signal'] and
                current_price > latest['bb_upper']
            )
            
            if long_signal:
                return 'long', current_price
            elif short_signal:
                return 'short', current_price
            
            return None, current_price
        
        except Exception as e:
            self.logger.error(f"Error generating trading signals for {symbol}: {e}")
            return None, None

    def setup_telegram(self):
        """Initialize Telegram bot"""
        self.telegram_token = 'TELEGRAM_TOKEN'
        self.telegram_chat_id = 'TELEGRAM_CHAT_ID'
        self.telegram_bot = telegram.Bot(token=self.telegram_token)
        self.logger.info("Telegram notifications initialized!")

    def send_telegram_message(self, message):
        """Send message to Telegram"""
        try:
            self.telegram_bot.send_message(
                chat_id=self.telegram_chat_id,
                text=message,
                parse_mode='HTML'
            )
        except Exception as e:
            self.logger.error(f"Failed to send Telegram message: {e}")    

    def send_advanced_notification(self, message_type, details):
        """
        Send advanced, formatted Telegram notifications
        
        Args:
            message_type (str): Type of notification (trade_open, trade_close, alert)
            details (dict): Notification details
        """
        message_templates = {
            'trade_open': (
                "🚀 New Trade Opened\n"
                "Symbol: {symbol}\n"
                "Type: {type}\n"
                "Entry Price: ${entry_price:.2f}\n"
                "Position Size: {position_size}"
            ),
            'trade_close': (
                "✅ Trade Closed\n"
                "Symbol: {symbol}\n"
                "Type: {type}\n"
                "Entry Price: ${entry_price:.2f}\n"
                "Exit Price: ${exit_price:.2f}\n"
                "Profit/Loss: ${profit_loss:.2f} ({profit_pct:.2f}%)"
            )
        }
        
        try:
            formatted_message = message_templates[message_type].format(**details)
            self.send_telegram_message(formatted_message)
        except Exception as e:
            self.logger.error(f"Telegram notification error: {e}")        


    def setup_logging(self):
        """Set up logging configuration"""
        if not hasattr(self, 'logger'):
            self.logger = logging.getLogger('BinanceTradingBot')
            self.logger.setLevel(logging.INFO)
            
            if self.logger.handlers:
                for handler in self.logger.handlers[:]:
                    self.logger.removeHandler(handler)
            
            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            
            # File handler
            file_handler = RotatingFileHandler(
                'logs/trading_bot.log', 
                maxBytes=10*1024*1024,
                backupCount=5
            )
            file_handler.setLevel(logging.INFO)
            
            # Formatters
            console_format = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
            file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s: %(message)s')
            
            console_handler.setFormatter(console_format)
            file_handler.setFormatter(file_format)
            
            self.logger.addHandler(console_handler)
            self.logger.addHandler(file_handler)

    def analyze_signal_strength(self, symbol, timeframe='15m'):
        """
        Analyze current market conditions and signal strength
        Returns a score between 0 (weakest) and 1 (strongest)
        """
        try:
            # Fetch recent data
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=100)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # Calculate indicators
            macd = ta.trend.MACD(close=df['close'])
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            df['macd_hist'] = macd.macd_diff()
            
            bollinger = ta.volatility.BollingerBands(close=df['close'])
            df['bb_upper'] = bollinger.bollinger_hband()
            df['bb_middle'] = bollinger.bollinger_mavg()
            df['bb_lower'] = bollinger.bollinger_lband()
            
            # Get latest values
            latest = df.iloc[-1]
            prev = df.iloc[-2]
            
            # Calculate individual strength factors
            score = 0.0
            
            # 1. MACD Strength (0-0.3)
            if latest['macd_hist'] > prev['macd_hist']:
                score += 0.3
            elif latest['macd_hist'] > 0:
                score += 0.15
                
            # 2. Bollinger Band Position (0-0.3)
            price_position = (latest['close'] - latest['bb_lower']) / (latest['bb_upper'] - latest['bb_lower'])
            if 0.3 <= price_position <= 0.7:  # Price in middle range
                score += 0.3
            elif 0.1 <= price_position <= 0.9:  # Price not at extremes
                score += 0.15
                
            # 3. Volume Analysis (0-0.2)
            volume_ma = df['volume'].rolling(20).mean().iloc[-1]
            if latest['volume'] > volume_ma:
                score += 0.2
                
            # 4. Trend Analysis (0-0.2)
            df['ma_20'] = df['close'].rolling(20).mean()
            df['ma_50'] = df['close'].rolling(50).mean()
            
            if (df['ma_20'].iloc[-1] > df['ma_20'].iloc[-2] and 
                df['ma_50'].iloc[-1] > df['ma_50'].iloc[-2]):
                score += 0.2
            
            return score
            
        except Exception as e:
            self.logger.error(f"Error analyzing signal strength for {symbol}: {e}")
            return 0.5  # Return neutral score on error    

            
    def execute_trade(self, symbol, trade_type, current_price):
        try:
            if symbol in self.open_trades:
                self.logger.info(f"Already have an open position for {symbol}")
                return

            balance = self.exchange.fetch_balance()
            total_balance = balance['total']['USDT']
            trade_amount = total_balance * self.trade_allocation_percentage
            position_size = (trade_amount * self.max_leverage) / current_price

            self.exchange.set_leverage(self.max_leverage, symbol)
            
            if trade_type == 'long':
                order = self.exchange.create_market_buy_order(symbol, position_size)
                stop_loss_price = current_price * (1 - self.stop_loss_percentage)
                take_profit_price = current_price * (1 + self.take_profit_percentage)
            else:
                order = self.exchange.create_market_sell_order(symbol, position_size)
                stop_loss_price = current_price * (1 + self.stop_loss_percentage)
                take_profit_price = current_price * (1 - self.take_profit_percentage)

            # Create trade details dictionary
            trade_details = {
                'type': trade_type,
                'entry_price': current_price,
                'stop_loss': stop_loss_price,
                'take_profit': take_profit_price,
                'position_size': position_size,
                'order_id': order['id'],
                'symbol': symbol,
                'entry_time': time.time()
            }

            # Add to open trades
            self.open_trades[symbol] = trade_details

            # Start a dedicated monitoring thread for this trade
            monitor_thread = threading.Thread(
                target=self.monitor_trade_in_real_time, 
                args=(symbol,), 
                daemon=True
            )
            monitor_thread.start()

            # Store the thread reference
            self.trade_monitoring_threads[symbol] = monitor_thread

            trade_msg = (
                f"New {trade_type.upper()} Trade Executed\n"
                f"Symbol: {symbol}\n"
                f"Entry: ${current_price:.2f}\n"
                f"Stop Loss: ${stop_loss_price:.2f}\n"
                f"Take Profit: ${take_profit_price:.2f}"
            )

            self.send_telegram_message(trade_msg)
            self.logger.info(f"New {trade_type} position opened for {symbol}")

        except Exception as e:
            self.logger.error(f"Error executing {trade_type} trade for {symbol}: {e}")

    def monitor_trade_in_real_time(self, symbol):
        """
        Continuously monitor a specific trade in real-time
        Checks price every second and manages stop loss and take profit
        """
        while symbol in self.open_trades:
            try:
                # Fetch current price
                ticker = self.exchange.fetch_ticker(symbol)
                current_price = ticker['last']

                # Get trade details
                trade = self.open_trades[symbol]

                # Determine if position should be closed
                should_close = False
                close_reason = ""

                if trade['type'] == 'long':
                    if current_price <= trade['stop_loss']:
                        should_close = True
                        close_reason = "Stop Loss Triggered"
                    elif current_price >= trade['take_profit']:
                        should_close = True
                        close_reason = "Take Profit Reached"
                else:  # short
                    if current_price >= trade['stop_loss']:
                        should_close = True
                        close_reason = "Stop Loss Triggered"
                    elif current_price <= trade['take_profit']:
                        should_close = True
                        close_reason = "Take Profit Reached"

                # Close position if conditions met
                if should_close:
                    self.close_position(symbol, trade, current_price)
                    
                    # Send notification
                    close_msg = (
                        f"Position Closed 🚨\n"
                        f"Symbol: {symbol}\n"
                        f"Reason: {close_reason}\n"
                        f"Current Price: ${current_price:.2f}"
                    )
                    self.send_telegram_message(close_msg)
                    break

                # Wait for 1 second before next check
                time.sleep(1)

            except Exception as e:
                self.logger.error(f"Error monitoring trade for {symbol}: {e}")
                time.sleep(1)


    def check_and_close_positions(self):
        """Advanced position management with dynamic stop loss and take profit"""
        for symbol in list(self.open_trades.keys()):
            try:
                # Get current price and market data
                ticker = self.exchange.fetch_ticker(symbol)
                current_price = ticker['last']
                
                trade = self.open_trades[symbol]
                
                # Calculate current profit percentage
                if trade['type'] == 'long':
                    current_profit_pct = ((current_price - trade['entry_price']) / trade['entry_price']) * 100
                else:  # short
                    current_profit_pct = ((trade['entry_price'] - current_price) / trade['entry_price']) * 100
                
                # Dynamic Stop Loss and Take Profit Management
                updated_trade = self.manage_trade_parameters(
                    symbol, 
                    trade, 
                    current_price, 
                    current_profit_pct
                )
                
                # Check if position should be closed
                should_close = self.should_close_position(
                    updated_trade, 
                    current_price, 
                    current_profit_pct
                )
                
                if should_close:
                    self.close_position(symbol, updated_trade, current_price)
                
                # Update trade in open trades
                self.open_trades[symbol] = updated_trade
            
            except Exception as e:
                self.logger.error(f"Error managing position for {symbol}: {e}")

    def manage_trade_parameters(self, symbol, trade, current_price, current_profit_pct):
        """
        Dynamic management of trade parameters based on profit levels and market conditions
        """
        try:
            # Get signal strength
            signal_strength = self.analyze_signal_strength(symbol)
            
            # Check position duration
            position_duration = time.time() - trade.get('entry_time', time.time())
            if position_duration > self.max_position_time:
                trade['force_close'] = True
                return trade
            
            # Initial phase (0-1% profit)
            if current_profit_pct >= self.breakeven_profit_level and current_profit_pct < self.min_take_profit:
                if trade['type'] == 'long':
                    trade['stop_loss'] = max(trade['entry_price'], trade['stop_loss'])
                else:
                    trade['stop_loss'] = min(trade['entry_price'], trade['stop_loss'])
            
            # Trailing phase (>1% profit)
            elif current_profit_pct >= self.min_take_profit:
                # Determine trailing distance based on profit level
                trailing_distance = self.get_trailing_distance(current_profit_pct)
                
                if trade['type'] == 'long':
                    new_stop_loss = current_price * (1 - trailing_distance)
                    trade['stop_loss'] = max(trade['stop_loss'], new_stop_loss)
                else:
                    new_stop_loss = current_price * (1 + trailing_distance)
                    trade['stop_loss'] = min(trade['stop_loss'], new_stop_loss)
                
                # Adjust take profit based on signal strength
                if signal_strength > 0.7:  # Strong signal
                    trade['take_profit'] = trade['entry_price'] * (1 + self.max_take_profit)
                else:  # Weak signal
                    trade['take_profit'] = current_price * (1 + self.min_take_profit)
            
            # Force close at maximum profit target
            if current_profit_pct >= self.max_take_profit:
                trade['force_close'] = True
            
            return trade
            
        except Exception as e:
            self.logger.error(f"Error managing trade parameters: {e}")
            return trade

    def get_trailing_distance(self, current_profit_pct):
        """
        Get appropriate trailing stop distance based on current profit level
        """
        for profit_level, distance in sorted(self.trailing_stop_distances.items(), reverse=True):
            if current_profit_pct >= profit_level:
                return distance
        return self.initial_stop_loss_percentage

    def should_close_position(self, trade, current_price, current_profit_pct):
        """
        Enhanced position closing logic
        """
        try:
            # Get current signal strength
            signal_strength = self.analyze_signal_strength(trade['symbol'])
            
            # Check force close flag
            if trade.get('force_close', False):
                return True
            
            # Check stop loss
            if trade['type'] == 'long':
                stop_loss_triggered = current_price <= trade['stop_loss']
            else:
                stop_loss_triggered = current_price >= trade['stop_loss']
            
            # Check take profit with signal strength consideration
            if current_profit_pct >= self.min_take_profit:
                if signal_strength < 0.3:  # Very weak signal
                    return True
                elif signal_strength < 0.5 and current_profit_pct >= (self.min_take_profit * 1.5):
                    return True
            
            return stop_loss_triggered
            
        except Exception as e:
            self.logger.error(f"Error in should_close_position: {e}")
            return False

    def close_position(self, symbol, trade, current_price):
        """
        Modified close_position method to handle real-time monitoring
        """
        try:
            # Close the position
            if trade['type'] == 'long':
                self.exchange.create_market_sell_order(symbol, trade['position_size'])
            else:
                self.exchange.create_market_buy_order(symbol, trade['position_size'])
            
            # Calculate profit/loss
            if trade['type'] == 'long':
                profit_loss = (current_price - trade['entry_price']) * trade['position_size']
            else:
                profit_loss = (trade['entry_price'] - current_price) * trade['position_size']
            
            # Log trade details
            trade_log = {
                'symbol': symbol,
                'type': trade['type'],
                'entry_price': trade['entry_price'],
                'exit_price': current_price,
                'profit_loss': profit_loss,
                'profit_percentage': ((current_price - trade['entry_price']) / trade['entry_price']) * 100
            }
            
            self.trade_history.append(trade_log)
            
            # Remove from open trades
            if symbol in self.open_trades:
                del self.open_trades[symbol]
            
            # Remove monitoring thread reference
            if symbol in self.trade_monitoring_threads:
                del self.trade_monitoring_threads[symbol]
        
        except Exception as e:
            self.logger.error(f"Error closing position for {symbol}: {e}")

    

    def trading_loop(self):
        """Main trading loop"""
        while self.is_running:
            try:
                for symbol in self.symbols:
                    for current_timeframe in self.timeframes:
                        # Generate trading signals
                        trade_signal, current_price = self.generate_trading_signals(symbol, current_timeframe)

                        if trade_signal:
                            signal_msg = (
                                f"Signal Generated 🎯\n"
                                f"Symbol: {symbol}\n"
                                f"Timeframe: {current_timeframe}\n"
                                f"Signal: {trade_signal.upper()}\n"
                                f"Price: ${current_price:.2f}"
                            )
                            self.send_telegram_message(signal_msg)
                        
                        # Execute trade if signal is present
                        if trade_signal:
                            self.execute_trade(symbol, trade_signal, current_price)
                
                # Check and close existing positions
                self.check_and_close_positions()
                
                # Display market status periodically
                self.display_market_status()
                
                # Wait before next iteration
                time.sleep(self.check_interval)
                
            except Exception as e:
                self.logger.error(f"Error in trading loop: {e}")
                error_msg = f"⚠️ Trading Error: {e}"
                self.send_telegram_message(error_msg)
                time.sleep(self.check_interval)

    def start_trading(self):
        """Start the trading bot"""
        if not self.is_running:
            self.is_running = True
            self.trading_thread = threading.Thread(target=self.trading_loop)
            self.trading_thread.start()
            start_msg = "Trading bot started!"
            self.logger.info(start_msg)
            self.send_telegram_message(start_msg)

    def stop_trading(self):
        """Stop the trading bot"""
        self.is_running = False
        if self.trading_thread:
            self.trading_thread.join()
        stop_msg = "Trading bot stopped!"
        self.logger.info(stop_msg)
        self.send_telegram_message(stop_msg)

    def display_market_status(self):
        """Display detailed market status, indicators, and open positions."""
        print("\n" + "=" * 100)
        print(" TRADING BOT STATUS ")
        print("=" * 100)

        # Display market analysis for each symbol and timeframe
        print("\nMarket Analysis:")
        headers = ['Symbol', 'Timeframe', 'Price', 'MACD', 'BB Status', 'Signal']
        market_data = []

        for symbol in self.symbols:
            for timeframe in self.timeframes:  # Iterate over defined timeframes
                try:
                    # Get trading signals and indicators
                    ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=100)
                    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

                    # Calculate indicators
                    macd = ta.trend.MACD(close=df['close'])
                    df['macd'] = macd.macd()
                    df['macd_signal'] = macd.macd_signal()
                    bollinger = ta.volatility.BollingerBands(close=df['close'])
                    df['bb_upper'] = bollinger.bollinger_hband()
                    df['bb_lower'] = bollinger.bollinger_lband()

                    latest = df.iloc[-1]
                    current_price = latest['close']

                    # Determine BB status
                    if current_price > latest['bb_upper']:
                        bb_status = "Above Upper"
                    elif current_price < latest['bb_lower']:
                        bb_status = "Below Lower"
                    else:
                        bb_status = "Inside"

                    # Determine trading signal
                    signal = "WAIT"
                    if (latest['macd'] > latest['macd_signal'] and 
                        current_price < latest['bb_lower']):
                        signal = "LONG"
                    elif (latest['macd'] < latest['macd_signal'] and 
                          current_price > latest['bb_upper']):
                        signal = "SHORT"

                    market_data.append([
                        symbol,
                        timeframe,  # Include the timeframe in the output
                        f"${current_price:.2f}",
                        f"{(latest['macd']-latest['macd_signal']):.3f} ({'Bullish' if latest['macd'] > latest['macd_signal'] else 'Bearish'})",
                        bb_status,
                        signal
                    ])
                except Exception as e:
                    self.logger.error(f"Error getting market data for {symbol} on {timeframe}: {e}")
                    market_data.append([symbol, timeframe, "Error", "Error", "Error", "Error"])

        print(tabulate.tabulate(market_data, headers=headers, tablefmt='grid'))

        # Display open positions
        if self.open_trades:
            print("\nOpen Positions:")
            position_headers = ['Symbol', 'Type', 'Entry', 'Current', 'Stop Loss', 'Take Profit', 'P/L']
            positions = []

            for symbol, trade in self.open_trades.items():
                try:
                    current_price = self.exchange.fetch_ticker(symbol)['last']
                    pl = (current_price - trade['entry_price']) if trade['type'] == 'long' else (trade['entry_price'] - current_price)

                    positions.append([
                        symbol,
                        trade['type'],
                        f"${trade['entry_price']:.2f}",
                        f"${current_price:.2f}",
                        f"${trade['stop_loss']:.2f}",
                        f"${trade['take_profit']:.2f}",
                        f"${pl:.2f}"
                    ])
                except Exception as e:
                    self.logger.error(f"Error getting position info for {symbol}: {e}")

            print(tabulate.tabulate(positions, headers=position_headers, tablefmt='grid'))
        else:
            print("\nNo open positions")

        # Display strategy parameters
        print("\nStrategy Parameters:")
        print(f"Take Profit: {self.take_profit_percentage * 100}%")
        print(f"Stop Loss: {self.stop_loss_percentage * 100}%")
        print(f"Leverage: {self.max_leverage}x")

        print("=" * 100 + "\n")


    
# [Rest of the code remains the same]

if __name__ == '__main__':
    # Create and run the trading bot
    bot = BinanceTradingBot()

    # Train ML model and analyze it before starting trading
    signal_generator = AdvancedSignalGenerator(bot.exchange)
    for symbol in bot.symbols:
        signal_generator.train_model(symbol, evaluate_performance=True)
    signal_generator.analyze_feature_importance()
    
    try:
        bot.start_trading()
        
        # Keep the main thread running
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nStopping the bot...")
        bot.stop_trading()
        print("Bot stopped successfully!")
