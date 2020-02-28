import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import pickle
from matplotlib import style
import time
import Blob

#ggplot:make the table to plot
style.use("ggplot")

SIZE = 10
#how many episodes
HM_EPISODES = 1000
MOVE_PENALTY = 1
ENEMY_PENALTY = 300
FOOD_REWARD = 25

epsilon = 0.9
EPS_DECAY = 0.9998
#how often show
SHOW_EVERY = 100

start_q_table = None #or filename

LEARNING_RATE = 0.1
DISCOUNT = 0.95

PLAYER_N = 1
FOOD_N = 2
ENEMY_N = 3

#set up the color
d = {
    1:(255, 175, 0), #Gritish blue
    2:(0, 255, 0),   #Green
    3:(0, 0, 255)    #Red
    }


if start_q_table is None:
    q_table = {}
    for x1 in range(-SIZE+1, SIZE):
        for y1 in range(-SIZE+1, SIZE):
            for x2 in range(-SIZE+1, SIZE):
                for y2 in range(-SIZE+1, SIZE):
                    q_table[((x1, y1), (x2, y2))] = [np.random.uniform(-5, 0) for i in range(4)]
else:
    with open(start_q_table, "rb") as f: #rb means read the binary file
        q_table = pickle.load(f)

for episode in range(HM_EPISODES):
    player = Blob.Blob()
    food = Blob.Blob()
    enemy = Blob.Blob()

    episode_reward = []
    if episode % SHOW_EVERY == 0:
        print(f"on #{episode}, epsilon:{epsilon}")
        #np.mean shows the average of elements each episode
        #episode_rewards[-SHOW_EVERY:] means picking up character string from -SHOW_EVERY to last
        # print(f"{SHOW_EVERY} episode reward average {np.mean(episode_rewards[-SHOW_EVERY:])}")
        show = True
    else:
        show = False

    episode_reward = []
    for i in range(200):
        print(type(food))
        #observation
        obs = (player - food, player - enemy)
        if np.random.random() > epsilon:
            action = np.argmax(q_table[obs])
        else:
            action = np.random.randint(0, 4)

        player.action(action)

    #enemy.move()
    #food.move()

        if player.x == enemy.x and player.y == enemy.y:
            reward = -ENEMY_PENALTY

            episode_reward.append(reward)


        elif player.x == food.x and player.y == food.y:
            reward = FOOD_REWARD

            episode_reward.append(reward)


        else:
            reward = -MOVE_PENALTY
            episode_reward.append(reward)



    new_obs = (player - food, player - enemy)
    max_future_q = np.max(q_table[new_obs])
    current_q = q_table[obs][action]

    if reward == FOOD_REWARD:
        new_q = FOOD_REWARD
    elif reward == -ENEMY_PENALTY:
        new_q = -ENEMY_PENALTY
    else:
        new_q =(1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)

    q_table[obs][action] = new_q

    if show:
        env = np.zeros((SIZE, SIZE, 3), dtype = np.uint8)  #3 means BGR
        env[player.y][player.x] = d[PLAYER_N]
        env[food.y][food.x] = d[FOOD_N]
        env[enemy.y][enemy.x] = d[ENEMY_N]

        img = Image.fromarray(env, "RGB")
        img = img.resize((300, 300))
        cv2.imshow("", np.array(img))

        if reward == FOOD_REWARD or reward == ENEMY_PENALTY:
            #0xFF = 11111111
            if cv2.waitKey(500) & 0xFF == ord("q"):
                break
            else:
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break


    epsilon *= EPS_DECAY

#convolve means 畳み込み積分, valid means not results when agent can't 畳み込み積分
'''There are always ValueError from below sentence. I think the code with np.ones is maybe wrong'''
print(episode_reward)
moving_avg = np.convolve(episode_reward, np.ones((SHOW_EVERY,)) / SHOW_EVERY, mode = "valid")

plt.plot([i for i in range(len(moving_avg))], moving_avg)
plt.ylabel(f"reward {SHOW_EVERY}ma")
plt.xlabel("episode #")
plt.show()

