from __future__ import print_function
import gym
import tensorflow as tf
import numpy as np
from itertools import count
from replay_memory import ReplayMemory, Transition
import tensorflow.contrib.layers as layers
import env_wrappers
import random
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--eval', action="store_true", default=False, help='Run in eval mode')
parser.add_argument('--seed', type=int, default=1, help='Random seed')
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)

class DQN(object):
    """
    A starter class to implement the Deep Q Network algorithm

    TODOs specify the main areas where logic needs to be added.

    If you get an error a Box2D error using the pip version try installing from source:
    > git clone https://github.com/pybox2d/pybox2d
    > pip install -e .

    """

    def __init__(self, env):

        self.env = env
        self.sess = tf.Session()

        # A few starter hyperparameters
        self.batch_size = 128
        self.gamma = 0.99
        # If using e-greedy exploration
        self.eps_start = 0.9
        self.eps_end = 0.05
        self.eps_decay = 1000 # in episodes
        # If using a target network
        self.clone_steps = 5000

        # memory
        self.replay_memory = ReplayMemory(100000)
        # Perhaps you want to have some samples in the memory before starting to train?
        self.min_replay_size = 512#10000

        self.replace_target_iter = 300
        self.curr_target_iter = 0
        self.lr = 0.01

        # define yours training operations here...
        self.observation_ph = tf.placeholder(tf.float32, shape=[None] + list(self.env.observation_space.shape))
        self.action_ph = tf.placeholder(tf.int32, shape=[None,])
        self.reward_ph = tf.placeholder(tf.float32, shape=[None,])
        self.next_observation_ph = tf.placeholder(tf.float32, shape=[None] + list(self.env.observation_space.shape))

        q_values = self.build_model(self.observation_ph)

        # define your update operations here...

        self.num_episodes = 0
        self.num_steps = 0

        self.saver = tf.train.Saver(tf.trainable_variables())
        self.sess.run(tf.global_variables_initializer())

    def build_model(self, observation_ph, scope='train'):
        """
        TODO: Define the tensorflow model

        Hint: You will need to define and input placeholder and output Q-values

        Currently returns an op that gives all zeros.
        """
        # eval net
        with tf.variable_scope('eval_net'):
            x = layers.fully_connected(observation_ph, 30, activation_fn=tf.nn.relu)
            self.q_eval_net = layers.fully_connected(x, self.env.action_space.n, activation_fn=None)

        # target net
        with tf.variable_scope('target_net'):
            x = layers.fully_connected(self.next_observation_ph, 30, activation_fn=tf.nn.relu)
            self.q_target_net = layers.fully_connected(x, self.env.action_space.n, activation_fn=None)

        with tf.variable_scope('q_target'):
            q_target = self.reward_ph + self.gamma * tf.reduce_max(self.q_target_net, axis=1, name='Qmax_s_')    # shape=(None, )
            self.q_target = tf.stop_gradient(q_target)
        with tf.variable_scope('q_eval'):
            a_indices = tf.stack([tf.range(tf.shape(self.action_ph)[0], dtype=tf.int32), self.action_ph], axis=1)
            self.q_eval_wrt_a = tf.gather_nd(params=self.q_eval_net, indices=a_indices)    # shape=(None, )
        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval_wrt_a, name='TD_error'))
        with tf.variable_scope('train'):
            self.train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')
        with tf.variable_scope('soft_replacement'):
            self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

    def select_action(self, obs, evaluation_mode=False):
        """
        TODO: Select an action given an observation using your model. This
        should include any exploration strategy you wish to implement

        If evaluation_mode=True, then this function should behave as if training is
        finished. This may be reducing exploration, etc.

        Currently returns a random action.
        """
        if self.num_episodes % self.eps_decay and self.eps_start > self.eps_end:
            self.eps_start -= 0.05
        obs = obs.reshape((-1, 8))
        if np.random.uniform() > self.eps_start or evaluation_mode:
            actions_value = self.sess.run(self.q_eval_net, feed_dict={self.observation_ph: obs})
            #print(np.argmax(actions_value))
            return np.argmax(actions_value)
        else:
            return env.action_space.sample()

    def update(self):
        """
        TODO: Implement the functionality to update the network according to the
        Q-learning rule
        """
        # Replace Target Network
        if self.curr_target_iter % self.replace_target_iter == 0:
            self.sess.run(self.target_replace_op)

        self.curr_target_iter += 1

        # Experience replay
        mem_pool = self.replay_memory.sample(self.batch_size)
        obs = np.reshape([_.state for _ in mem_pool], (-1, 8))
        next_obs = np.reshape([_.next_state for _ in mem_pool], (-1, 8))
        acts = [_.action for _ in mem_pool]
        rewards = [_.reward for _ in mem_pool]

        self.sess.run([self.train_op, self.loss], 
            feed_dict={self.observation_ph: obs, 
            self.action_ph: acts, 
            self.reward_ph: rewards,
            self.next_observation_ph: next_obs
        })

        #raise NotImplementedError

    def train(self):
        """
        The training loop. This runs a single episode.

        TODO: Implement the following as desired:
            1. Storing transitions to the ReplayMemory
            2. Updating the network at some frequency
            3. Backing up the current parameters to a reference, target network
        """
        done = False
        obs = env.reset()
        while not done:
            action = self.select_action(obs, evaluation_mode=False)
            next_obs, reward, done, info = env.step(action)
            self.num_steps += 1
            self.replay_memory.push(obs, action, next_obs, reward, done)

            if self.num_steps >= self.min_replay_size:
                self.update()
        self.num_episodes += 1

    def eval(self, save_snapshot=True):
        """
        Run an evaluation episode, this will call
        """
        total_reward = 0.0
        ep_steps = 0
        done = False
        obs = env.reset()
        while not done:
            env.render()
            action = self.select_action(obs, evaluation_mode=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward
        print ("Evaluation episode: ", total_reward)
        if save_snapshot:
            print ("Saving state with Saver")
            self.saver.save(self.sess, 'models/dqn-model', global_step=self.num_episodes)

def train(dqn):
    for i in count(1):
        dqn.train()
        # every 10 episodes run an evaluation episode
        if i % 10 == 0:
            dqn.eval()

def eval(dqn):
    """
    Load the latest model and run a test episode
    """
    ckpt_file = os.path.join(os.path.dirname(__file__), 'models/checkpoint')
    with open(ckpt_file, 'r') as f:
        first_line = f.readline()
        model_name = first_line.split()[-1].strip("\"")
    dqn.saver.restore(dqn.sess, os.path.join(os.path.dirname(__file__), 'models/'+model_name))
    dqn.eval(save_snapshot=False)


if __name__ == '__main__':
    # On the LunarLander-v2 env a near-optimal score is some where around 250.
    # Your agent should be able to get to a score >0 fairly quickly at which point
    # it may simply be hitting the ground too hard or a bit jerky. Getting to ~250
    # may require some fine tuning.
    env = gym.make('LunarLander-v2')
    env.seed(args.seed)
    # Consider using this for the challenge portion
    # env = env_wrappers.wrap_env(env)

    dqn = DQN(env)
    if args.eval:
        eval(dqn)
    else:
        train(dqn)
