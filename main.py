from env import TradingEnv


if __name__ == "__main__":
    tradingenv = TradingEnv()
    state = tradingenv.reset()
    print("Initial State:", state)

    c = 0
    while True:
        c += 1
        action = tradingenv.action_space.sample()  # Sample random action
        # print("ACTION: ", action[0])
        state, reward, done, info = tradingenv.step(action)
        # tradingenv.render()
        if c % 1000 == 0:
            print(f"{c}")
        if done:
            break
