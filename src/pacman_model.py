import random

from keras.layers import TimeDistributed, Dense, RepeatVector, recurrent, Input
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
import keras.backend as K

import numpy as np

from .tensorboard_utils import add_summary


class PacmanModel(object):
    def __init__(self, max_enemies=15, hidden_size=128, batch_size=4096, invert=True, optimizer_lr=0.001, clipnorm=None, logdir=None):
        # To be edited by Qi. Add the variables required by the RL agent
        self.max_enemies = max_enemies #Don't delete this
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.invert = invert
        self.optimizer_lr = optimizer_lr #Learning Rate
        self.clipnorm = clipnorm # Used to define the optimizer for the neural network keras model
        self.logdir = logdir #Don't delete this

        self.epochs = 0
        self.make_model()
        if logdir:
            self.callbacks = [TensorBoard(log_dir=self.logdir)]
        else:
            self.callbacks = []

    def make_model(self): # To be filled in by Qi. Define the neural network for the RL agent.
        self.model = 1 #Save the neural network in self.model

    def generate_data(self, dist, size):
        problem_set = []
        while len(problem_set) <= size:
            problem_set.append(1 + np.random.choice(len(dist), p=dist))

        return problem_set

    def train_epoch(self, train_data, val_data=None): # To be filled in by qi
        # Train RL agent on train_data, save reward in 'reward'. Train RL agent on val_data, save reward in 'val_reward'
        # Return reward and val_reward
        reward = []
        val_reward = []

        return reward, val_reward


class PacmanEnvironment:
    def __init__(self, model, train_size, val_size, val_dist, writer=None):
        self.model = model
        self.num_subtasks = model.max_enemies
        self.train_size = train_size
        self.val_data = model.generate_data(val_dist, val_size)
        self.writer = writer

    def step(self, train_dist):
        print("Training on", train_dist)
        train_data = self.model.generate_data(train_dist, self.train_size)
        reward, val_reward = self.model.train_epoch(train_data, self.val_data)

        train_done = False
        val_done = False

        # if self.writer:
        #     for k, v in history.items():
        #         add_summary(self.writer, "model/" + k, v[-1], self.model.epochs)
        #     for i in range(self.num_actions):
        #         # add_summary(self.writer, "train_accuracies/task_%d" % (i + 1), train_accs[i], self.model.epochs)
        #         add_summary(self.writer, "valid_accuracies/task_%d" % (i + 1), val_accs[i], self.model.epochs)

        return reward, train_done, val_done
