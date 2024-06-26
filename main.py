from env import TradingEnv


if __name__ == "__main__":
    tradingenv = TradingEnv()
    state = tradingenv.reset()
    print("Initial State:", state)

    for _ in range(10):
        action = tradingenv.action_space.sample()  # Sample random action
        print("ACTION: ", action[0])
        state, reward, done, info = tradingenv.step(action[0])
        tradingenv.render()
        if done:
            break
