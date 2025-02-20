import os
import gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import optuna

# For RL training with Stable Baselines 3
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv

from trading_env import TradingEnv, load_clean_data
from config import (
    CSV_PATH,
    TRANSACTION_COST,
    NOISE_STD,
    MIN_EPISODE_LENGTH,
    MAX_TRADES_PER_DAY,
    MAX_HOLD_MINUTES,
    INITIAL_ACCOUNT_BALANCE,
    TRADE_PENALTY,
    MIN_HOLD_STEPS,
    MAX_TRADES_PER_EPISODE,
    DEFAULT_SELL_BONUS_THRESHOLD,
    DEFAULT_SELL_BONUS_MULTIPLIER,
    DEFAULT_SELL_BONUS_THRESHOLD_TIER2,
    DEFAULT_SELL_BONUS_MULTIPLIER_TIER2,
    DEFAULT_SELL_BONUS_MULTIPLIER_TIER3,
    DEFAULT_SELL_BONUS_THRESHOLD_TIER3,
    DEFAULT_SELL_REWARD_SCALING,
    DEFAULT_HOLD_REWARD_SCALING,
    TOTAL_TIMESTEPS,
    TUNING_TRIALS,
    TUNING_TOTAL_TIMESTEPS,
    N_EPISODES_EVALUATION,
    TUNING_N_EPISODES_EVALUATION
)

# ---------------------------------------------
# RL Training & Evaluation Functions
# ---------------------------------------------

def train_rl_agent(env, total_timesteps):
    model = PPO("MlpPolicy", env, verbose=1, device="cpu")
    model.learn(total_timesteps=total_timesteps)
    return model

def evaluate_rl_agent(model, env, n_episodes=N_EPISODES_EVALUATION, accuracy_weight=0.1, trade_count_weight=10):
    episode_rewards = []
    episode_final_balances = []
    all_accuracies = []
    all_avg_profit_per_trade = []
    episode_trade_counts = []

    for episode in range(n_episodes):
        obs = env.reset()
        done = False
        total_reward = 0.0
        starting_balance = env.account_balance

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward

        daily_balances = info.get('daily_balances', {})
        final_balance = env.account_balance

        trade_profits = info.get('trade_profits', [])
        num_trades = len(trade_profits)
        episode_trade_counts.append(num_trades)

        if num_trades > 0:
            positive_trades = sum(1 for p in trade_profits if p > 0)
            accuracy = (positive_trades / num_trades) * 100
            avg_profit_per_trade = np.mean(trade_profits)
        else:
            accuracy = 0.0
            avg_profit_per_trade = 0.0

        all_accuracies.append(accuracy)
        all_avg_profit_per_trade.append(avg_profit_per_trade)
        episode_rewards.append(total_reward)
        episode_final_balances.append(final_balance)

        print(f"\n=== Episode {episode+1} Summary ===")
        print(f"Starting Balance: {starting_balance:.2f}")
        for day in sorted(daily_balances.keys()):
            daily_bal = daily_balances[day]
            print(f"  {day}: Balance = {daily_bal:.2f}")
        print(f"Final Balance: {final_balance:.2f}")
        print(f"Total Reward: {total_reward:.2f}")
        print(f"Total Trades: {num_trades}")
        print(f"Accuracy (profitable trades): {accuracy:.2f}%")
        print(f"Average Profit per Trade (Trade Size): {avg_profit_per_trade:.4f}")
        if num_trades > 0:
            print(f"Largest Trade: {max(trade_profits):.4f}")
        else:
            print("Largest Trade: 0.0000")
        print("====================================\n")

    overall_avg_accuracy = np.mean(all_accuracies)
    overall_avg_trade_size = np.mean(all_avg_profit_per_trade)
    overall_avg_reward = np.mean(episode_rewards)
    overall_avg_trades = np.mean(episode_trade_counts)

    composite_score = overall_avg_reward + (accuracy_weight * overall_avg_accuracy) - (trade_count_weight * overall_avg_trades)

    print(f"Average Reward over {n_episodes} Episodes: {overall_avg_reward:.2f}")
    print(f"Average Final Balance over {n_episodes} Episodes: {np.mean(episode_final_balances):.2f}")
    print(f"Average Accuracy over {n_episodes} Episodes: {overall_avg_accuracy:.2f}%")
    print(f"Average Trade Size over {n_episodes} Episodes: {overall_avg_trade_size:.4f}")
    print(f"Average Number of Trades per Episode: {overall_avg_trades:.2f}")
    print(f"Composite Score: {composite_score:.2f}")

    return composite_score

def make_env():
    def _init():
        df = load_clean_data(CSV_PATH)
        return TradingEnv(
            df,
            transaction_cost=TRANSACTION_COST,
            noise_std=NOISE_STD,
            min_episode_length=MIN_EPISODE_LENGTH,
            max_trades_per_day=MAX_TRADES_PER_DAY,
            max_hold_minutes=MAX_HOLD_MINUTES,
            initial_balance=INITIAL_ACCOUNT_BALANCE,
            trade_penalty=TRADE_PENALTY,
            min_hold_steps=MIN_HOLD_STEPS,
            max_trades_per_episode=MAX_TRADES_PER_EPISODE,
            sell_bonus_threshold=DEFAULT_SELL_BONUS_THRESHOLD,
            sell_bonus_multiplier=DEFAULT_SELL_BONUS_MULTIPLIER,
            sell_bonus_threshold_tier2=DEFAULT_SELL_BONUS_THRESHOLD_TIER2,
            sell_bonus_multiplier_tier2=DEFAULT_SELL_BONUS_MULTIPLIER_TIER2,
            sell_bonus_multiplier_tier3=DEFAULT_SELL_BONUS_MULTIPLIER_TIER3,
            sell_bonus_threshold_tier3=DEFAULT_SELL_BONUS_THRESHOLD_TIER3,
            sell_reward_scaling=DEFAULT_SELL_REWARD_SCALING,
            hold_reward_scaling=DEFAULT_HOLD_REWARD_SCALING
        )
    return _init

def objective(trial):
    sell_bonus_multiplier = trial.suggest_float("SELL_BONUS_MULTIPLIER", 50, 200)
    sell_bonus_threshold = trial.suggest_float("SELL_BONUS_THRESHOLD", 1.0, 2.0)
    sell_bonus_multiplier_tier2 = trial.suggest_float("SELL_BONUS_MULTIPLIER_TIER2", 100, 300)
    sell_bonus_threshold_tier2 = trial.suggest_float("SELL_BONUS_THRESHOLD_TIER2", 2.0, 4.0)
    sell_bonus_multiplier_tier3 = trial.suggest_float("SELL_BONUS_MULTIPLIER_TIER3", 500, 1000)
    sell_bonus_threshold_tier3 = trial.suggest_float("SELL_BONUS_THRESHOLD_TIER3", 4.0, 6.0)

    accuracy_weight = trial.suggest_float("accuracy_weight", 0.05, 0.2)
    trade_count_weight = trial.suggest_float("trade_count_weight", 5, 15)

    sell_reward_scaling = trial.suggest_float("sell_reward_scaling", 50, 200)
    hold_reward_scaling = trial.suggest_float("hold_reward_scaling", 5, 20)

    print(f"Trial parameters: SELL_BONUS_MULTIPLIER={sell_bonus_multiplier}, SELL_BONUS_THRESHOLD={sell_bonus_threshold}, "
          f"SELL_BONUS_MULTIPLIER_TIER2={sell_bonus_multiplier_tier2}, SELL_BONUS_THRESHOLD_TIER2={sell_bonus_threshold_tier2}, "
          f"SELL_BONUS_MULTIPLIER_TIER3={sell_bonus_multiplier_tier3}, SELL_BONUS_THRESHOLD_TIER3={sell_bonus_threshold_tier3}, "
          f"accuracy_weight={accuracy_weight}, trade_count_weight={trade_count_weight}, "
          f"sell_reward_scaling={sell_reward_scaling}, hold_reward_scaling={hold_reward_scaling}")

    df = load_clean_data(CSV_PATH)
    env = TradingEnv(
        df,
        transaction_cost=TRANSACTION_COST,
        noise_std=NOISE_STD,
        min_episode_length=MIN_EPISODE_LENGTH,
        max_trades_per_day=MAX_TRADES_PER_DAY,
        max_hold_minutes=MAX_HOLD_MINUTES,
        initial_balance=INITIAL_ACCOUNT_BALANCE,
        trade_penalty=TRADE_PENALTY,
        min_hold_steps=MIN_HOLD_STEPS,
        max_trades_per_episode=MAX_TRADES_PER_EPISODE,
        sell_bonus_threshold=sell_bonus_threshold,
        sell_bonus_multiplier=sell_bonus_multiplier,
        sell_bonus_threshold_tier2=sell_bonus_threshold_tier2,
        sell_bonus_multiplier_tier2=sell_bonus_multiplier_tier2,
        sell_bonus_multiplier_tier3=sell_bonus_multiplier_tier3,
        sell_bonus_threshold_tier3=sell_bonus_threshold_tier3,
        sell_reward_scaling=sell_reward_scaling,
        hold_reward_scaling=hold_reward_scaling
    )

    model = PPO("MlpPolicy", env, verbose=0, device="cpu")
    model.learn(total_timesteps=TUNING_TOTAL_TIMESTEPS)

    avg_reward = evaluate_rl_agent(
        model, env, n_episodes=TUNING_N_EPISODES_EVALUATION,
        accuracy_weight=accuracy_weight,
        trade_count_weight=trade_count_weight
    )
    return avg_reward

if __name__ == "__main__":
    df = load_clean_data(CSV_PATH)
    num_envs = 4
    envs = SubprocVecEnv([make_env() for _ in range(num_envs)])
    model = PPO("MlpPolicy", envs, verbose=1, device="cpu")
    model = train_rl_agent(envs, total_timesteps=TOTAL_TIMESTEPS)
    model.save(os.path.join(os.getcwd(), "latest_trading_bot_model"))

    single_env = TradingEnv(
        df,
        transaction_cost=TRANSACTION_COST,
        noise_std=NOISE_STD,
        min_episode_length=MIN_EPISODE_LENGTH,
        max_trades_per_day=MAX_TRADES_PER_DAY,
        max_hold_minutes=MAX_HOLD_MINUTES,
        initial_balance=INITIAL_ACCOUNT_BALANCE,
        trade_penalty=TRADE_PENALTY,
        min_hold_steps=MIN_HOLD_STEPS,
        max_trades_per_episode=MAX_TRADES_PER_EPISODE,
        sell_bonus_threshold=DEFAULT_SELL_BONUS_THRESHOLD,
        sell_bonus_multiplier=DEFAULT_SELL_BONUS_MULTIPLIER,
        sell_bonus_threshold_tier2=DEFAULT_SELL_BONUS_THRESHOLD_TIER2,
        sell_bonus_multiplier_tier2=DEFAULT_SELL_BONUS_MULTIPLIER_TIER2,
        sell_bonus_multiplier_tier3=DEFAULT_SELL_BONUS_MULTIPLIER_TIER3,
        sell_bonus_threshold_tier3=DEFAULT_SELL_BONUS_THRESHOLD_TIER3,
        sell_reward_scaling=DEFAULT_SELL_REWARD_SCALING,
        hold_reward_scaling=DEFAULT_HOLD_REWARD_SCALING
    )
    evaluate_rl_agent(model, single_env, n_episodes=N_EPISODES_EVALUATION)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=TUNING_TRIALS)

    print("\nTuning completed. Best trial:")
    trial = study.best_trial
    print("  Value (Average Reward): {:.2f}".format(trial.value))
    print("  Parameters:")
    for key, value in trial.params.items():
        print("    {}: {:.2f}".format(key, value))
