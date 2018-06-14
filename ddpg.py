import chainer.functions as F
from chainer import optimizers, Variable, serializers
from collections import deque
import gym
import matplotlib.pyplot as plt
import numpy as np
import sys

from network import NN

SEED_NUMPY = 12345
np.random.seed(SEED_NUMPY)


def ou_process(initial_state):
    """Ornstein–Uhlenbeck process.

    Args:
    initial_state (ndarray): Initial state of the process.

    Yields:
    state (ndarray): Yields current state value for each step.

    Note:
    http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
    """
    x = initial_state
    dimension = len(x)

    theta = 0.6
    sigma = 0.3
    mu = 0.0

    disc = 1
    while True:
        yield x
        x += disc * (theta * (mu - x) + sigma * np.random.normal(size=dimension))


def log(log_list):
    if len(log_list) == 0:
        return ''
    else:
        log = ''.join(log_list)
        return log


class Actor(object):
    def __init__(self, n_st, n_act):
        super(Actor, self).__init__()
        self.n_st = n_st
        self.n_act = n_act
        self.model = NN(n_st, n_act)
        self.optimizer = optimizers.Adam()
        self.optimizer.setup(self.model)
        self.noise = ou_process(np.zeros((n_act), dtype=np.float32))

    def action(self, st, noise=False):
        a = self.model(st, norm=True)

        if noise:
            n = next(self.noise)
            a = np.clip(a.data + n, -1, 1)
            return a
        else:
            return a.data

    def update(self, st, dqda):
        mu = self.model(st, norm=True)
        self.model.cleargrads()
        mu.grad = -dqda
        mu.backward()
        self.optimizer.update()

    def update_target(self, tau, current_NN):
        self.model.weight_update(tau, current_NN)

    def save_model(self, outputfile):
        serializers.save_npz(outputfile, self.model)

    def load_model(self, inputfile):
        serializers.load_npz(inputfile, self.model)


class Critic(object):
    def __init__(self, n_st, n_act):
        super(Critic, self).__init__()
        self.n_st = n_st
        self.n_act = n_act
        self.model = NN(n_st + n_act, 1)
        self.optimizer = optimizers.Adam()
        self.optimizer.setup(self.model)
        self.log = []

    def Q_value(self, st, act):
        state_action_vector = np.concatenate((st, act), axis=1)
        Q = self.model(state_action_vector).data
        return Q

    def return_dqda(self, st, act):
        state_action_vector = Variable(np.concatenate((st, act), axis=1))
        self.model.cleargrads()
        Q = self.model(state_action_vector)
        Q.grad = np.ones((state_action_vector.shape[0], 1), dtype=np.float32)
        Q.backward()
        grad = state_action_vector.grad[:, self.n_st:]
        return grad

    def update(self, y, st, act):
        self.model.cleargrads()

        state_action_vector = np.concatenate((st, act), axis=1)
        Q = self.model(state_action_vector)

        loss = F.mean_squared_error(Q, Variable(y))

        loss.backward()
        self.optimizer.update()

        self.log.append('Q:{0},y:{1}\n'.format(Q.data.T, y.T))

        return loss.data

    def update_target(self, tau, current_NN):
        self.model.weight_update(tau, current_NN)

    def save_model(self, outputfile):
        serializers.save_npz(outputfile, self.model)

    def load_model(self, inputfile):
        serializers.load_npz(inputfile, self.model)


class DDPG(object):
    def __init__(self, n_st, n_act, tau=0.001):
        super(DDPG, self).__init__()
        sys.setrecursionlimit(10000)
        self.n_st = n_st
        self.n_act = n_act

        self.actor = Actor(n_st, n_act)
        self.actor_target = Actor(n_st, n_act)
        self.actor_target.update_target(1.0, self.actor.model)
        self.critic = Critic(n_st, n_act)
        self.critic_target = Critic(n_st, n_act)
        self.critic_target.update_target(1.0, self.critic.model)

        self.memory = deque()
        self.memory_size = 1000
        self.tau = tau
        self.loss = 0
        self.step = 0
        self.gamma = 0.99
        self.batch_size = 200
        self.train_freq = 10
        self.log = []

    def stock_experience(self, st, act, r, st_dash, ep_end):
        self.memory.append((st, act, r, st_dash, ep_end))
        if len(self.memory) > self.memory_size:
            self.memory.popleft()

    def shuffle_memory(self):
        mem = np.array(self.memory)
        return np.random.permutation(mem)

    def parse_batch(self, batch):
        st, act, r, st_dash, ep_end = [], [], [], [], []
        for i in range(self.batch_size):
            st.append(batch[i][0])
            act.append(batch[i][1])
            r.append(batch[i][2])
            st_dash.append(batch[i][3])
            ep_end.append(batch[i][4])
        st = np.array(st, dtype=np.float32)
        act = np.array(act, dtype=np.float32)
        r = np.array(r, dtype=np.float32)
        st_dash = np.array(st_dash, dtype=np.float32)
        ep_end = np.array(ep_end, dtype=np.bool)
        return st, act, r, st_dash, ep_end

    def learn(self, st, act, r, st_dash, ep_end):
        a_dash_target = self.actor_target.action(st_dash)

        Q_target = self.critic_target.Q_value(st_dash, a_dash_target)

        y = r + self.gamma * Q_target * (1 - ep_end)
        y = y.astype(np.float32)

        loss = self.critic.update(y, st, act)

        self.actor.update(st, self.critic.return_dqda(st, self.actor.action(st)))

        self.critic_target.update_target(self.tau, self.critic.model)
        self.actor_target.update_target(self.tau, self.actor.model)

        self.loss += loss

        self.log.append(log(self.critic.log))

    def get_action(self, st):
        s = Variable(st)
        if self.step < 1000:
            a = self.actor.action(s, noise=True)
        else:
            a = self.actor.action(s)
        return a[0][0]

    def experience_replay(self):
        mem = self.shuffle_memory()
        perm = np.array(range(len(mem)))

        index = perm[0:self.batch_size]
        batch = mem[index]
        st, act, r, st_dash, ep_end = self.parse_batch(batch)
        self.learn(st, act, r, st_dash, ep_end)

    def train(self):
        if len(self.memory) >= self.memory_size:
            if self.step % self.train_freq == 0:
                self.experience_replay()
        self.step += 1

    def save_model(self):
        self.actor.save_model('actor_network.model')
        self.critic.save_model('critic_network.model')
        self.actor_target.save_model('actor_target_network.model')
        self.critic_target.save_model('critic_target_network.model')

    def load_model(self):
        self.actor.load_model('actor_network.model')
        self.critic.load_model('critic_network.model')
        self.actor_target.load_model('actor_target_network.model')
        self.critic_target.load_model('critic_target_network.model')


def main_train():
    env_name = "CartPole-v0"
    seed = 0
    env = gym.make(env_name)

    n_st = env.observation_space.shape[0]

    if type(env.action_space) == gym.spaces.discrete.Discrete:
        # CartPole-v0, Acrobot-v0, MountainCar-v0
        n_act = env.action_space.n
        action_list = np.arange(0, n_act)
    elif type(env.action_space) == gym.spaces.box.Box:
        # Pendulum-v0
        action_list = [np.array([a]) for a in [-2.0, 2.0]]
        n_act = len(action_list)

    agent = DDPG(n_st, 1, seed)
    # env.Monitor.start(view_path, video_callable=None, force=True, seed=seed)

    list_t = []
    log = []
    for i_episode in range(1000):
        print("episode_num" + str(i_episode))
        observation = env.reset()
        for t in range(200):
            env.render()
            state = observation.astype(np.float32)
            state_reshape = state.reshape((1, n_st))
            act_i = agent.get_action(state_reshape)

            if act_i < 0:
                action = action_list[0]
            else:
                action = action_list[1]

            observation, reward, ep_end, _ = env.step(action)
            state_dash = observation.astype(np.float32)

            if action == 0:
                action = -1
            else:
                action = 1

            agent.stock_experience(state, [action], [reward], state_dash, [ep_end])

            agent.train()
            print(log([1, 2]))
            log_tmp = 'i_episode:{0},act_i:{1},action:{2},loss:{3},{4}\n'.format(i_episode, act_i, action, agent.loss, log(agent.log))
            log.append(log_tmp)

            if ep_end:
                print("max t:" + str(t))
                list_t.append(t)
                break

    with open('log.txt', "w") as f:
        log = ''.join(log)
        f.write(log)
    agent.save_model()

    plt.plot(list_t)
    plt.show()
    # env.Monitor.close()
    return agent


if __name__ == "__main__":
    agent = main_train()
