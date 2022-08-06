#!/usr/bin/env python3
import numpy as np
import rospy
from sensor_msgs.msg import JointState
from math import pi


cartpole_state = np.array([0, 0, 0, 0])


def cartpole_state(msg: JointState):  
    global cartpole_state  
    theta, theta_dot = msg.position[0], msg.velocity[0]
    pos_x, vel_x = msg.position[1], msg.velocity[1]
    
    theta += pi
    while theta < 0:
        theta += 2 * pi
    while theta > 2 * pi:
        theta -= 2 * pi
    if theta > pi:
        theta -= 2 *pi
    
    cartpole_state = [pos_x, vel_x, theta, theta_dot]



class CartPoleAgent:
    def __init__(self):
        np.random.seed(42)

        # Environment Variables
        self.actions = [-32, 32]

        # Q table and buckets
        # [position, velocity, angle, angular velocity]
        self.buckets = (1, 1, 24, 24)
        self.upper_bounds = [15, 200, np.pi, 11]
        self.lower_bounds = [-15, -200, -np.pi, -11]
        self.Qtable = np.zeros(self.buckets + (len(self.actions),))

        # Qlearning
        self.n_episodes = 10000
        self.gamma = 0.98
        self.min_lr = 0.1
        self.decay_rate = 200.

    def qlearning_train(self):
        print("start trianing")
        for e in range(self.n_episodes):
            state = self.__discretize_state(self.env.reset())
            alpha = exploration_rate = self.__rate(e)
            done = False
            while not done:
                self.env.render()
                # Choose action based on ep_greedy_policy
                action = self.__choose_action(exploration_rate, state)
                print(cartpole_state)
                # get new state, reward, and end-signal
                new_state, reward, done, _ = self.env.step(action)
                new_state = self.__discretize_state(new_state)
                # Update Q table
                self.__update_q(state, action, reward, alpha, new_state)
                state = new_state
        print("finished training")

    def __choose_action(self, exploration_rate, state):
        if np.random.random() < exploration_rate:
            # Samples a random action given the environment, returns a positive or negative force
            state_rand = np.random.random()
            if state_rand < 0.5:
                return self.actions[0]
            else:
                return self.actions[1]
        else:
            return np.argmax(self.Qtable[state])

    def __rate(self, e):
        return max(self.min_lr, min(1., 1. - np.log10((e + 1) / self.decay_rate)))

    def __discretize_state(self, state):
        # The upper and the lower bounds for the discretization
        discretized = list()
        for i in range(len(state)):
            scaling = (state[i] + abs(self.lower_bounds[i])) / (self.upper_bounds[i] - self.lower_bounds[i])
            new_obs = int(round((self.buckets[i] - 1) * scaling))
            new_obs = min(self.buckets[i] - 1, max(0, new_obs))
            discretized.append(new_obs)
        return tuple(discretized)

    def __update_q(self, state, action, r, a, new_state):
        self.Qtable[state][action] = self.Qtable[state][action] + \
                                     a * (r + self.gamma * np.max(self.Qtable[new_state]) - self.Qtable[state][action])
                                     
    def train(self):
        while not rospy.is_shutdown():
            self.cartpole_state = cartpole_state
            rospy.sleep(0.01)


if __name__ == '__main__':
    rospy.init_node("cartpole_q")
    sub = rospy.Subscriber("/cart_pole/joint_states", JointState, callback=cartpole_state)    
    agent = CartPoleAgent()
    agent.train()
