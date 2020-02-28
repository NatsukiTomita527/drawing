import numpy as np
import numba
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import pickle
from matplotlib import style
import time
import csv
import Blob
import BLACK
import WHITE



im = cv2.imread("smile_test.png")
style.use("ggplot")

h, w, _ = im.shape
SIZE_x = w
SIZE_y = h

start_q_table = None #or filename
HM_EPISODES = 1000
MOVE_PENALTY = 1
BLACK_REWARD = 25
WHITE_PENALTY = 300

epsilon = 0.9
EPS_DECAY = 0.9998
SHOW_EVERY = 100

start_q_table = None

LEARNING_RATE = 0.1
DISCOUNT = 0.95

AGENT_N = 1
BLACK_N = 2
WHITE_N = 3

d = {
    1:(0, 0, 255),       #Red
    2:(0, 0, 0),         #Black
    3:(255, 255, 255)    #White
    }



def table(w, h):
    q_table = {}
    for x1 in range(-w + 1, w):
        for y1 in range(-h + 1, h):

            # array = np.array([np.array([x1, y1], [x2, y2])])
            # array_uniform = np.array([np.random.uniform(-5, 0) for i in range(4)])
            # print("a")
            q_table[(x1, y1)] = [np.random.uniform(-5, 0, 4)]
    return q_table


q_table = table(w, h)




# # else:
# #     with open(start_q_table, "rb") as f: #rb means read the binary file
# #         q_table = pickle.load(f)
#
# print("dupa")
#
for episode in range(HM_EPISODES):
    agent = Blob.Blob()
    black_point = BLACK.Black()
    white_point = WHITE.White()
    episode_reward = []

    if episode % SHOW_EVERY == 0:
        print(f"on #{episode}, epsilon:{epsilon}")
        #np.mean shows the average of elements each episode
        #episode_rewards[-SHOW_EVERY:] means picking up character string from -SHOW_EVERY to last
        # print(f"{SHOW_EVERY} episode reward average {np.mean(episode_rewards[-SHOW_EVERY:])}")
        show = True
    else:
        show = False
    obs_s = []


    episode_reward = []
    agent_plot = []
    image_out = np.ones(im.shape, np.uint8)*255
    image_out_array = np.zeros(im.shape)

    for i in range(100):

        #observation
        obs = (Blob.Blob.__sub__(agent, black_point))
        print(obs)

        if np.random.random() > epsilon:
                action = np.argmax(q_table[obs])

        else:
                action = np.random.randint(0, 4)
        print(action)
        agent.action(action)
        # image_out_array[agent.x][agent.y] = 1
        agent_plot.append([agent.x, agent.y])
        print(agent_plot)
        count_black = 0
        count_white = 0

        if agent.x == black_point.x and agent.y == black_point.y:
            reward = BLACK_REWARD
            count_black += 1

        elif agent.x == white_point.x and agent.y == white_point.y:
            reward = -WHITE_PENALTY
            count_white += 1

        else:
            reward = -MOVE_PENALTY


        if count_black >=500:
            print("black")
            break

        elif count_white >=3000:
            print("white")
            break


        new_obs =  (Blob.Blob.__sub__(agent, black_point))
        max_future_q = np.max(q_table[new_obs])
        current_q_array = q_table[obs][0]
        print(current_q_array)
        current_q = current_q_array[action]
        print(current_q)
        print(type(current_q))

        #current_q = current_q_array[action]

        REWARD=0
        if reward == BLACK_REWARD:
            new_q = BLACK_REWARD
            REWARD += reward

        elif reward == -WHITE_PENALTY:
            new_q = -WHITE_PENALTY
            REWARD += reward
        else:
            print(current_q)
            new_q =(1 - LEARNING_RATE) * (current_q) + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            REWARD += reward

        print(REWARD)
        current_q_array[action] = new_q

    if show:
        env = np.zeros((SIZE_x, SIZE_y, 3), dtype = np.uint8)  #3 means BGR
        env[agent.x][agent.y] = d[AGENT_N]
        env[black_point.x][black_point.y] = d[BLACK_N]
        env[white_point.x][white_point.y] = d[WHITE_N]

        img = Image.fromarray(env, "RGB")
        img = img.resize((300, 300))
        cv2.imshow("", np.array(img))
        if reward == BLACK_REWARD or reward == WHITE_PENALTY:
            if cv2.waitKey(500) & 0xFF == ord("q"):
                break
            else:
                if cv2.waitKey(1) & 0xFF == ord("q"):

                    break

    epsilon *= EPS_DECAY



# moving_avg = np.convolve(episode_reward, np.ones((SHOW_EVERY,)) / SHOW_EVERY, mode = "valid")
# plt.plot([i for i in range(len(moving_avg))], moving_avg)
# plt.ylabel(f"reward {SHOW_EVERY}ma")
# plt.xlabel("episode #")
# plt.show()
plt.plot(agent_plot)
plt.show()
# for agent.y in range(0, h):              #transform array to image       1->[0, 0, 0]    0->[255, 255, 255]
#     for agent.x in range(0, w):
#         if all(image_out_array[agent.x, agent.y]) == True:
#             image_out[agent.x, agent.y] = [0, 0, 0]
#             d = d + [[agent.x, agent.y]]
#             print(d[-1])
#             # with open('main.csv', 'w') as file:
#             #     writer = csv.writer(file, lineterminator='\n')
#             #     writer.writerows(d)
#
#
#
#         else:
#             image_out[agent.x, agent.y] = [255, 255, 255]
#
# cv2.imshow("output image", image_out)                          #displaying out image