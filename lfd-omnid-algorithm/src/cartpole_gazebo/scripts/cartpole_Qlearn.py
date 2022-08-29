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
    """
    Gets the current cartpole state from the cartpole system from the node /cart_pole/joint_states
    - Updates cartpole_state global variable to the current state

    :param msg: The current joint state from the model (JointState)
    :return: None
    """
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
        # Position state vector x: [position, velocity, angle, angular velocity]
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
        """
        Function called to train the Qlearning agent
        Calls in 2 threads:
        - Update state of cartpole system to current state
        - Qlearning training

        :return: None
        """
        update_state_thread = self.update_state()
        qlearning_thread = self.qlearning()
        qlearning_thread.join()
        update_state_thread.join()

    @Decorators.thread_decorator
    def qlearning(self):
        """
        QLearning training agent

        :return: None
        """
        print("Start Training")
        for e in range(self.n_episodes):
            # Gets the current discretized state of the updated cartpole state
            state = self.__discretize_state(self.cartpole_state)
            alpha = exploration_rate = self.__rate(e)
            done = False
            ii = 0

            while not done:
                # Choose action based on ep_greedy_policy
                action = self.__choose_action(exploration_rate, state)
                pub.publish(self.actions[action])
                time.sleep(0.01)
                # get new state, reward, and end-signal
                new_state = self.cartpole_state[:]
                done, reward = self.check_state(new_state, ii)
                new_state = self.__discretize_state(new_state)
                # Update Q table
                self.__update_q(state, action, reward, alpha, new_state)
                state = new_state
                ii += 1

            pub.publish(0)
            rospy.wait_for_service('/gazebo/reset_simulation')
            reset_world = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
            reset_world()
            time.sleep(2)
        self.update_flag = False
        print("finished training")

    def __choose_action(self, exploration_rate, state):
        """
        Samples a random action given the environment, returns a positive or negative force

        :param exploration_rate: Current exploration rate of the system (float)
        :param state: Current discritized state - position vector of state (np array of shape (1, 4))
        :return: Action - Force applied to system (int)
        """
        if np.random.random() < exploration_rate:
            state_rand = np.random.random()
            if state_rand < 0.5:
                return 0
            else:
                return 1
        else:
            return np.argmax(self.Qtable[state])

    def __rate(self, e):
        """
        Current exploration state

        :param e: Episode number (int)
        :return: Exploration Rate (float)
        """
        return max(self.min_lr, min(1., 1. - np.log10((e + 1) / self.decay_rate)))

    def __discretize_state(self, state):
        """
        Takes the state and discritizes it using scaled bins

        :param state: Current discritized state of the cartpole system - position vector of x (np array (1, 4))
        :return: Discritized position vector state (tuple)
        """
        discretized = list()
        for i in range(len(state)):
            scaling = (state[i] + abs(self.lower_bounds[i])) / (self.upper_bounds[i] - self.lower_bounds[i])
            new_obs = int(round((self.buckets[i] - 1) * scaling))
            new_obs = min(self.buckets[i] - 1, max(0, new_obs))
            discretized.append(new_obs)
        return tuple(discretized)

    def __update_q(self, state, action, r, a, new_state):
        """
        Updates q_learning matrix

        :param state: Current discritized state of the cartpole system - position vector of x (np array (1, 4))
        :param action: Action taken by the agent - Force (Float)
        :param r: Current reward earned for system - Reward (Float)
        :param a: The exploration rate for the system (Float)
        :param new_state: New discritized state of the system - position vector of x (np array (1, 4))
        :return: None
        """
        self.Qtable[state][action] = self.Qtable[state][action] + \
                                     a * (r + self.gamma * np.max(self.Qtable[new_state]) - self.Qtable[state][action])

    @Decorators.thread_decorator
    def update_state(self):
        """
        Gets the next state from the cartpole system
        - Updates self.cartpole_state
        :return: None
        """
        while not rospy.is_shutdown() and self.update_flag:
            self.cartpole_state = cartpole_state
            rospy.sleep(0.1)

    def check_state(self, new_state, ii):
        """
        Checks the current position state of the cartpole system

        :param new_state: New cartpole state as a position vector (np array of shape (1, 4))
        :param iter: Iteration of Q learning step
        :return: Termination, Reward (Tuple)
                Termination - Whether the Q learning Agent should terminate the current episode
                Reward - Reward for current state
        """
        print(f"state: {new_state}, iter: {ii}")
        x, xd, t, td = new_state[0], new_state[1], new_state[2], new_state[3]
        terminate = bool(x > 10 or x < -10 or xd > 5 or xd < -5 or td > 11 or td < -11 or ii > self.train_iter)
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
