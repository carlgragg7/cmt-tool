import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()

class TradingBot:
    def __init__(self, symbol):
        self.symbol = symbol
        self.data = self.fetch_data()
        self.ordered_emas = []

    def fetch_data(self):
        return yf.download(self.symbol, period='1y', interval='1d')

    def calculate_sma(self, window):
        self.data[f'SMA_{window}'] = self.data['Close'].rolling(window=window).mean()

    def calculate_ema(self, window):
        self.ordered_emas.append(window)
        self.ordered_emas.sort()
        self.data[f'EMA_{window}'] = self.data['Close'].ewm(span=window, adjust=False).mean()

    def calculate_bollinger_bands(self, window):
        self.data['BB_Middle'] = self.data['Close'].rolling(window).mean()
        self.data['BB_Upper'] = self.data['BB_Middle'] + 2*self.data['Close'].rolling(window).std()
        self.data['BB_Lower'] = self.data['BB_Middle'] - 2*self.data['Close'].rolling(window).std()

    def calculate_rsi(self, window):
        delta = self.data['Close'].diff()
        up_days = delta.copy()
        up_days[delta<=0]=0.0
        down_days = abs(delta.copy())
        down_days[delta>0]=0.0
        RS = up_days.rolling(window).mean()/down_days.rolling(window).mean()
        self.data['RSI'] = 100 - (100 / (1 + RS))

    def calculate_returns(self):
        self.data['Returns'] = self.data['Close'].pct_change()

    def calculate_macd(self, short_window, long_window):
        self.data['MACD_line'] = self.data['Close'].ewm(span=short_window, adjust=False).mean() - self.data['Close'].ewm(span=long_window, adjust=False).mean()
        self.data['Signal_line'] = self.data['MACD_line'].ewm(span=9, adjust=False).mean()

    def calculate_max_drawdown(self):
        self.data['Cumulative Returns'] = (1 + self.data['Returns']).cumprod()
        self.data['Running Max'] = self.data['Cumulative Returns'].cummax()
        self.data['Drawdown'] = self.data['Running Max'] - self.data['Cumulative Returns']
        self.data['Max Drawdown'] = self.data['Drawdown'].max()

    def calculate_moving_average_convergence_divergence(self, short_window, long_window):
        # Calculate the MACD line
        self.data['MACD_line'] = self.data['Close'].ewm(span=short_window, adjust=False).mean() - self.data['Close'].ewm(span=long_window, adjust=False).mean()
        # Calculate the signal line
        self.data['Signal_line'] = self.data['MACD_line'].ewm(span=9, adjust=False).mean()
        # Calculate the MACD histogram
        self.data['MACD_Histogram'] = self.data['MACD_line'] - self.data['Signal_line']

    def generate_signals(self):
        # Calculate the difference between the short-term and long-term moving averages
        self.data['MA_Difference'] = self.data[f'EMA_{self.ordered_emas[0]}'] - self.data[f'EMA_{self.ordered_emas[-1]}']

        # Generate a buy signal when the short-term moving average crosses above the long-term moving average
        self.data['Buy_Signal'] = np.where((self.data['MA_Difference'] > 0) & (self.data['MA_Difference'].shift() < 0), self.data['Close'], np.nan)

        # Generate a sell signal when the short-term moving average crosses below the long-term moving average
        self.data['Sell_Signal'] = np.where((self.data['MA_Difference'] < 0) & (self.data['MA_Difference'].shift() > 0), self.data['Close'], np.nan)

    def plot_signals(self):
        if 'Buy_Signal' not in self.data.columns or 'Sell_Signal' not in self.data.columns:
            print("No signals to plot. Have you run the generate_signals method?")
            return

        plt.figure(figsize=(12,5))
        plt.plot(self.data['Close'], label='Close Price', color='blue', alpha=0.35)
        plt.plot(self.data['BB_Upper'], label='Upper Bollinger Band', color='red', alpha=0.35)
        plt.plot(self.data['BB_Lower'], label='Lower Bollinger Band', color='green', alpha=0.35)
        plt.scatter(self.data.index, self.data['Buy_Signal'], color='green', marker='^', alpha=1)
        plt.scatter(self.data.index, self.data['Sell_Signal'], color='red', marker='v', alpha=1)
        plt.title(f'{self.symbol} Close Price, Buy & Sell Signals')
        plt.xlabel('Date')
        plt.ylabel('Close Price USD ($)')
        plt.legend(loc='upper left')
        plt.show()

    def simulate_trading(self):
        # Initialize account balance and positions
        account_balance = 10000.0
        positions = 0.0

        # Iterate through the data and execute trades based on the signals
        for i in range(1, len(self.data)):
            # Check for a buy signal
            if not np.isnan(self.data['Buy_Signal'].iloc[i]):
                # Calculate the number of shares to buy based on available funds
                price = self.data['Close'].iloc[i]
                shares = account_balance // price

                # Update account balance and positions
                account_balance -= shares * price
                positions += shares

                # Print the buy trade details
                print(f"Buy {shares} shares of {self.symbol} at ${price:.2f}")
                print(f"Account Balance: ${account_balance:.2f}")
                print(f"Change in Account Balance: ${(positions * price) - account_balance:.2f}")

            # Check for a sell signal
            if not np.isnan(self.data['Sell_Signal'].iloc[i]):
                # Calculate the number of shares to sell based on available positions
                price = self.data['Close'].iloc[i]

                # Update account balance and positions
                account_balance += positions * price
                positions = 0.0

                # Print the sell trade details
                print(f"Sell all shares of {self.symbol} at ${price:.2f}")
                print(f"Account Balance: ${account_balance:.2f}")
                print(f"Change in Account Balance: ${(positions * price) - account_balance:.2f}")

        # Print the final account balance
        print(f"Final Account Balance: ${account_balance:.2f}")
        print(f"Value of Positions: ${(positions * price):.2f}")

    def run(self):
        self.calculate_sma(20)
        self.calculate_ema(5)
        self.calculate_ema(10)
        self.calculate_ema(20)
        self.calculate_bollinger_bands(20)
        self.calculate_rsi(14)
        self.calculate_macd(12, 26)

        self.calculate_returns()
        self.calculate_max_drawdown()
        self.calculate_moving_average_convergence_divergence(12, 26)
        self.generate_signals()
        self.plot_signals()
        self.simulate_trading()

    def save_data_to_csv(self, filename):
        new_data = self.data.copy()
        # only the current data
        new_data = new_data.iloc[-1:]
        new_data['Date'] = new_data.index

        with open(filename, 'a') as f:
            new_data.to_csv(f, index=False, header=True, na_rep='')

while True:
    stock_symbol = input("Symbol: ")
    if stock_symbol == "quit":
        break
    bot = TradingBot(stock_symbol)
    bot.run()
    bot.save_data_to_csv(f'{bot.symbol}_data.csv')
