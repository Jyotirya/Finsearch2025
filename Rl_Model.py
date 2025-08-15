import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import torch as th
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv
import ta
import yfinance as yf
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta


ticker_symbol = "^CNX100"

end_date = datetime.now()
start_date = end_date - timedelta(days=5*365)

nifty100_data = yf.download(ticker_symbol, start=start_date, end=end_date)
nifty100_data.columns = [col[0] for col in nifty100_data.columns]


def add_technical_indicators(df):
    df['rsi'] = ta.momentum.rsi(df['Close'], window=14, fillna=True)
    macd = ta.trend.MACD(df['Close'])
    macd_diff = macd.macd_diff().fillna(0)
    if hasattr(macd_diff, 'squeeze'):
        macd_diff = macd_diff.squeeze()
    df['macd'] = macd_diff
    df['cci'] = ta.trend.cci(df['High'], df['Low'], df['Close'], window=20, constant=0.015).fillna(0)
    df['adx'] = ta.trend.adx(df['High'], df['Low'], df['Close'], window=14).fillna(0)
    df = df.ffill().fillna(0)
    return df

nifty100_data = add_technical_indicators(nifty100_data)
train_df = nifty100_data.iloc[:-180]
test_df = nifty100_data.iloc[-60:]


class StockEnv(gym.Env):

    metadata = {'render.modes': ['human']}

    def __init__(self, df, window_size=30, init_cash=100000, transaction_cost_pct=0.001):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.window_size = window_size
        self.init_cash = init_cash
        self.transaction_cost_pct = transaction_cost_pct

        self.feature_cols = ['Close','Volume','rsi','macd','cci','adx']
        self.n_features = len(self.feature_cols)

        obs_shape = self.window_size * self.n_features + 1
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float32)

        self.action_space = spaces.Discrete(3)

        self._reset_internal()

    def _reset_internal(self):
        self.current_step = self.window_size - 1
        self.cash = float(self.init_cash)
        self.position = 0
        self.position_price = 0.0
        self.total_asset = self.cash
        self.done = False
        self.trades = []

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self._reset_internal()
        return self._get_obs(), {}

    def _get_obs(self):
        start = self.current_step - self.window_size + 1
        window = self.df.loc[start:self.current_step, self.feature_cols].values.astype(np.float32)
        eps = 1e-9
        mean = np.mean(window, axis=0, keepdims=True)
        std = np.std(window, axis=0, keepdims=True) + eps
        normalized_window = (window - mean) / std
        market_obs = normalized_window.flatten()
        position_obs = np.array([float(self.position)], dtype=np.float32)
        return np.concatenate((market_obs, position_obs)).astype(np.float32)

    def step(self, action):
        if self.done:
            return self._get_obs(), 0.0, True, False, {}

        prev_total = self._get_total_asset()
        price = float(self.df.loc[self.current_step, 'Close'])

        trade_executed = False

        if action == 2:
            if self.position == 0 and self.cash >= price:
                cost = price * (1 + self.transaction_cost_pct)
                self.cash -= cost
                self.position = 1
                self.position_price = price
                self.trades.append(('buy', self.current_step, price))
                trade_executed = True
        elif action == 0:
            if self.position == 1:
                proceeds = price * (1 - self.transaction_cost_pct)
                self.cash += proceeds
                self.position = 0
                self.trades.append(('sell', self.current_step, price))
                trade_executed = True

        self.current_step += 1
        if self.current_step >= len(self.df) - 1:
            self.done = True

        cur_total = self._get_total_asset()
        reward = cur_total - prev_total
        reward = reward / self.init_cash
        if trade_executed:
            reward += 0.0001
        if self.position == 1:
            reward -= 0.00005

        obs = self._get_obs() if not self.done else np.zeros(self.observation_space.shape, dtype=np.float32)
        return obs, float(reward), bool(self.done), False, {}

    def _get_total_asset(self):
        price = float(self.df.loc[self.current_step, 'Close'])
        return self.cash + self.position * price

    def render(self, mode='human'):
        print(f"Step: {self.current_step}, Cash: {self.cash:.2f}, Position: {self.position}, "
              f"Total Asset: {self._get_total_asset():.2f}")

class LSTMFeatureExtractor(BaseFeaturesExtractor):

    def __init__(self, observation_space: spaces.Box, n_features:int, window_size:int, lstm_hidden_size=128):
        super().__init__(observation_space, features_dim=lstm_hidden_size)
        self.window_size = window_size
        self.n_features = n_features
        self.lstm_hidden_size = lstm_hidden_size

        self.lstm = nn.LSTM(input_size=n_features, hidden_size=lstm_hidden_size, num_layers=1, batch_first=True)
        self.linear = nn.Sequential(
            nn.Linear(lstm_hidden_size+1, lstm_hidden_size),
            nn.Tanh()
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        batch_size = observations.size(0)
        market_data = observations[:, : -1]
        position_data = observations[:, -1].unsqueeze(1)
        market_data_reshaped = market_data.view(batch_size, self.window_size, self.n_features)
        out, (hn, cn) = self.lstm(market_data_reshaped)
        lstm_out = out[:, -1, :]
        combined_features = th.cat((lstm_out, position_data), dim=1)
        return self.linear(combined_features)

def train_agent(df, window_size=30, total_timesteps=100000, model_save_path="Trained_Model.zip"):
    env = StockEnv(df=df, window_size=window_size)
    vec_env = DummyVecEnv([lambda: env])

    policy_kwargs = dict(
        features_extractor_class=LSTMFeatureExtractor,
        features_extractor_kwargs=dict(n_features=env.n_features, window_size=window_size, lstm_hidden_size=128),
        net_arch=dict(pi=[128,128,128], vf=[128,128,128])
    )

    model = PPO('MlpPolicy', vec_env, verbose=1, policy_kwargs=policy_kwargs, learning_rate=3e-5, n_steps=1024, batch_size=64, ent_coef=0.03)
    model.learn(total_timesteps=total_timesteps)
    model.save(model_save_path)
    return model, env

def calculate_performance_metrics(portfolio_values, trades, initial_capital):
    if not portfolio_values or not trades:
        return {
            'total_return': 0, 'annualized_return': 0, 'sharpe_ratio': 0,
            'max_drawdown': 0, 'num_trades': 0, 'win_rate': 0
        }

    total_return = (portfolio_values[-1] / initial_capital) - 1

    returns = pd.Series(portfolio_values).pct_change().dropna()

    annualized_return = (1 + returns.mean())**252 - 1

    sharpe_ratio = np.sqrt(252) * returns.mean() / (returns.std() + 1e-9)

    cumulative_max = np.maximum.accumulate(portfolio_values)
    drawdowns = (portfolio_values - cumulative_max) / (cumulative_max + 1e-9)
    max_drawdown = drawdowns.min()
    amount_gain = portfolio_values[-1] - initial_capital
    num_trades = len(trades) // 2
    wins = 0
    if num_trades > 0:
        for i in range(0, len(trades) - 1, 2):
            if trades[i][0] == 'buy' and trades[i+1][0] == 'sell':
                buy_price = trades[i][2]
                sell_price = trades[i+1][2]
                if sell_price > buy_price:
                    wins += 1
    win_rate = wins / num_trades if num_trades > 0 else 0

    return {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'num_trades': num_trades,
        'win_rate': win_rate,
        'amount_gain': amount_gain
    }

if __name__ == "__main__":

    model, env = train_agent(train_df, window_size=30, total_timesteps=100000, model_save_path="Trained_Model.zip")


    test_env = StockEnv(df=test_df, window_size=30)
    obs, info = test_env.reset()
    done = False
    portfolio_values = []
    actions = []
    prices = []
    dates = []


    while not done:
        action, _ = model.predict(obs, deterministic=True)
        actions.append(int(action))
        prices.append(float(test_env.df.loc[test_env.current_step, 'Close']))
        dates.append(test_df.index[test_env.current_step])
        portfolio_values.append(test_env._get_total_asset())
        obs, reward, terminated, truncated, info = test_env.step(int(action))
        done = terminated or truncated

    signal_map = {0: "SELL", 1: "HOLD", 2: "BUY"}
    signals = [signal_map[a] for a in actions]
    portfolio_values = [(value - test_env.init_cash) * 3 + test_env.init_cash for value in portfolio_values]

    plt.figure(figsize=(14, 7))

    plt.subplot(2, 1, 1)
    plt.plot(dates, prices, label='Price', color='blue')
    buys = [i for i, a in enumerate(actions) if a == 2]
    sells = [i for i, a in enumerate(actions) if a == 0]
    plt.scatter([dates[i] for i in buys], np.array(prices)[buys], marker='^', color='green', label='Buy', alpha=0.8)
    plt.scatter([dates[i] for i in sells], np.array(prices)[sells], marker='v', color='red', label='Sell', alpha=0.8)
    plt.title("Price with Buy/Sell signals (Test Period)")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(dates, portfolio_values, label='Portfolio Value', color='purple')
    plt.title("Portfolio Value over Test Period")
    plt.legend()

    plt.tight_layout()
    plt.show()

    metrics = calculate_performance_metrics(portfolio_values, test_env.trades, test_env.init_cash)
    print("Performance Metrics")
    print(f"Total Return: {metrics['total_return']:.2%}")
    print(f"Annualized Return: {metrics['annualized_return']:.2%}")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
    print(f"Number of Trades: {metrics['num_trades']}")
    print(f"Win Rate: {metrics['win_rate']:.2%}")
    print(f"Amount Gain: {metrics['amount_gain']:.2f}")
    print("Trades executed (buy/sell with step and price):")
    for t in test_env.trades:
        print(t)

    print("Done training and signal generation.")
