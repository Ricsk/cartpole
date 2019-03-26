import gym
from model import PolicyGradient
# from RL_brain import PolicyGradient
import matplotlib.pyplot as plt

DISPLAY_REWARD_THRESHOLD = -2000  # renders environment if total episode reward is greater then this threshold
# episode: 0   reward: -11586
# episode: 100   reward: -6995
# episode: 200   reward: -3644
# episode: 300   reward: -1798
# episode: 400   reward: -977
# episode: 500   reward: -738
# episode: 600   reward: -539
# episode: 700   reward: -608
# episode: 800   reward: -478
# episode: 900   reward: -751
RENDER = False  # rendering wastes time

env = gym.make('MountainCar-v0')
env.seed(1)     # reproducible, general Policy gradient has high variance
env = env.unwrapped

print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)

RL = PolicyGradient(
    n_actions=env.action_space.n,
    n_features=env.observation_space.shape[0],
    learning_rate=0.02,
    reward_decay=0.995,
    # output_graph=True,
)

for i_episode in range(1000):


    observation = env.reset()

    while True:
        if RENDER:
            env.render()

        action = RL.choose_action(observation)

        observation_, reward, done, info = env.step(action)     # reward = -1 in all cases

        RL.store_transition(observation, action, reward)

        if done:
            # calculate running reward
            ep_rs_sum = sum(RL.ep_rs)
            if 'running_reward' not in globals():
                running_reward = ep_rs_sum
            else:
                running_reward = running_reward * 0.99 + ep_rs_sum * 0.01
            if running_reward > DISPLAY_REWARD_THRESHOLD:
                RENDER = True     # rendering

            print("episode:", i_episode, "  reward:", int(running_reward))

            vt = RL.learn()  # train

            if i_episode == 30:
                plt.plot(vt)  # plot the episode vt
                plt.xlabel('episode steps')
                plt.ylabel('normalized state-action value')
                plt.show()

            break

        observation = observation_