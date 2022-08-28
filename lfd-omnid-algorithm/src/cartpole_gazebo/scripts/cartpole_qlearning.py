#!/usr/bin/env python3
import numpy as np
import rospy
from std_msgs.msg import Float64
from sensor_msgs.msg import JointState
from math import pi
import threading
import time
from std_srvs.srv import Empty
from gazebo_msgs.srv import GetModelState 


cartpole_state = [0, 0, pi, 0]


def cartpole_state_func(msg: JointState):  
    global cartpole_state  
    theta, theta_dot = msg.position[0], msg.velocity[0]
    pos_x, vel_x = msg.position[1], msg.velocity[1]
    
    theta += pi
    while theta < 0:
        theta += 2 * pi
    while theta > 2 * pi:
        theta -= 2 * pi
    if theta > pi:
        theta -= 2 * pi
    
    cartpole_state = [pos_x, vel_x, theta, theta_dot]


class CartPoleAgent:
    class Decorators(object):
        @classmethod
        def thread_decorator(cls, func):
            def func_thread(*args, **kwargs):
                t = threading.Thread(target=func, args=args, kwargs=kwargs)
                t.start()
                return t
            return func_thread

    def __init__(self):
        np.random.seed(42)

        # Environment Variables
        self.actions = [-64, 64]
        self.cartpole_state = cartpole_state

        # Q table and buckets
        # [position, velocity, angle, angular velocity]
        self.buckets = (1, 1, 12, 12)
        self.upper_bounds = [15, 200, np.pi, 11]
        self.lower_bounds = [-15, -200, -np.pi, -11]
        self.Qtable = np.zeros(self.buckets + (len(self.actions),))

        # Qlearning
        self.n_episodes = 100
        self.train_iter = 1000
        self.gamma = 0.98
        self.min_lr = 0.1
        self.decay_rate = 200.

        # Rospy integration
        self.update_flag = True

    def train(self):
        update_state_thread = self.update_state()
        qlearning_thread = self.qlearning()
        qlearning_thread.join()
        update_state_thread.join()

    @Decorators.thread_decorator
    def qlearning(self):
        print("start trianing")
        model_coordinates = rospy.ServiceProxy( '/gazebo/get_model_state', GetModelState)
        rospy.wait_for_service('/gazebo/get_model_state')
        print(model_coordinates("robot", "pole"))
        for e in range(self.n_episodes):
            print(e)
            state = self.__discretize_state(self.cartpole_state)
            alpha = exploration_rate = self.__rate(e)
            done = False
            iter = 0
            while not done:
                # Choose action based on ep_greedy_policy
                action = self.__choose_action(exploration_rate, state)
                pub.publish(self.actions[action])
                time.sleep(0.01)
                # get new state, reward, and end-signal
                new_state = self.cartpole_state[:]
                done, reward = self.check_state(new_state, iter)
                new_state = self.__discretize_state(new_state)
                # Update Q table
                self.__update_q(state, action, reward, alpha, new_state)
                state = new_state
                iter += 1
                
            pub.publish(0)
            rospy.wait_for_service('/gazebo/reset_simulation')
            reset_world = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
            reset_world()
            time.sleep(2)
        self.update_flag = False
        print("finished training")

    def __choose_action(self, exploration_rate, state):
        if np.random.random() < exploration_rate:
            # Samples a random action given the environment, returns a positive or negative force
            state_rand = np.random.random()
            if state_rand < 0.5:
                return 0
            else:
                return 1
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

    @Decorators.thread_decorator
    def update_state(self):
        while not rospy.is_shutdown() and self.update_flag:
            self.cartpole_state = cartpole_state
            rospy.sleep(0.1)

    def check_state(self, new_state, iter):
        print(f"state: {new_state}, iter: {iter}")
        x, xd, t, td = new_state[0], new_state[1], new_state[2], new_state[3]
        terminate = bool(x > 10 or x < -10 or xd > 5 or xd < -5 or td > 11 or td < -11 or iter > self.train_iter)
        if terminate:
            return terminate, 0.0
        reward = 1.0
        if abs(t) < 0.2:
            reward += 10
            if abs(td) < 0.2:
                reward += 100
        elif abs(t) < 0.4 and abs(td) < 0.4:
            reward += 20
        elif abs(t) < 1.0 and abs(td) < 3:
            reward += 1
        return terminate, reward


if __name__ == '__main__':
    rospy.init_node("cartpole_q")
    pub = rospy.Publisher("/cart_pole_controller/command", Float64, queue_size=1)
    sub = rospy.Subscriber("/cart_pole/joint_states", JointState, callback=cartpole_state_func)    
    agent = CartPoleAgent()
    agent.train()
