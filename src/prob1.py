import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
sns.set()
import time


class MacCormack:
    def __init__(self, xrange, trange, delta_x, delta_t):
        self.num_x = int(xrange / delta_x) + 1
        self.num_t = int(trange / delta_t) + 1
        self.Q = np.zeros((self.num_t, 3, self.num_x))
        self.E = np.zeros((self.num_t, 3, self.num_x))
        self.delta_x = delta_x
        self.delta_t = delta_t
        self.gamma = 1.4
        self.cnt = 0

    

    def set_init(self, left, right):
        pl, rhol, ul = left
        pr, rhor, ur = right
        self.Q[0][0, :self.num_x//2] = rhol
        self.Q[0][0, self.num_x//2:] = rhor
        self.Q[0][1, :self.num_x//2] = rhol * ul
        self.Q[0][1, self.num_x//2:] = rhor * ur
        self.Q[0][2, :self.num_x//2] = 0.5 * (rhol * ul**2) + pl / (self.gamma - 1)
        self.Q[0][2, self.num_x//2:] = 0.5 * (rhor * ur**2) + pr / (self.gamma - 1)
        self.E[0] = self.Q2E(self.Q[0])
    

    def Q2E(self, Q):
        E = np.zeros(Q.shape)
        p = (self.gamma - 1) * (Q[2] - 0.5 * Q[1] * Q[1] / (Q[0]+1e-9))
        E[0] = Q[1]
        E[1] = p + Q[1] * Q[1] / (Q[0]+1e-9)
        E[2] = (Q[2] + p) * Q[1] / (Q[0]+1e-9)
        return E

    
    def update(self):
        Q_star = np.zeros(self.Q[0].shape)
        Q_star[:, 1:] = self.Q[self.cnt][:, 1:] - self.delta_t / self.delta_x * (self.E[self.cnt][:, 1:] - self.E[self.cnt][:, :-1])
        Q_star[:, 0] = self.Q[self.cnt][:, 0]
        E_star = self.Q2E(Q_star)
        self.Q[self.cnt+1][:, 1:-1] = 0.5 * (self.Q[self.cnt][:, 1:-1] + Q_star[:, 1:-1]) - 0.5 * self.delta_t / self.delta_x * (E_star[:, 2:] - E_star[:, 1:-1])
        p = (self.gamma - 1) * (self.Q[self.cnt][2] - 0.5 * self.Q[self.cnt][1] * self.Q[self.cnt][1] / (self.Q[self.cnt][0]+1e-9))
        self.Q[self.cnt+1][:, 1:-1] += 1 * abs(p[2:] - 2 * p[1:-1] + p[:-2]) / abs(p[2:] + 2 * p[1:-1] + p[:-2]) * (self.Q[self.cnt][:, 2:] - 2 * self.Q[self.cnt][:, 1:-1] + self.Q[self.cnt][:, :-2])
        self.Q[self.cnt+1][:, 0] = self.Q[self.cnt][:, 0]
        self.Q[self.cnt+1][:, -1] = self.Q[self.cnt][:, -1]
        self.E[self.cnt+1] = self.Q2E(self.Q[self.cnt+1])
        self.cnt += 1

    
    def simulate(self):
        while self.cnt != self.num_t-1:
            self.update()
    

    def visualize(self, tlist):
        tlist = np.array(list(map(int, tlist/self.delta_t)))

        # rho
        fig = plt.figure(figsize=(10, 8))
        for t in tlist:
            rho = self.Q[t][0]
            plt.plot([self.delta_x * i for i in range(self.num_x)], rho, c=cm.winter(1 - t/tlist[-1]), label='t = {}'.format(t/1000000))
        plt.legend()
        plt.savefig('../output/prob1/rho.png', dpi=200, bbox_inches='tight')
        
        # p
        fig = plt.figure(figsize=(10, 8))
        for t in tlist:
            p = self.E[t][1] - self.Q[t][1] * self.Q[t][1] / self.Q[t][0]
            plt.plot([self.delta_x * i for i in range(self.num_x)], p, c=cm.winter(1 - t/tlist[-1]), label='t = {}'.format(t/1000000))
        plt.legend()
        plt.savefig('../output/prob1/p.png', dpi=200, bbox_inches='tight')

        # u
        fig = plt.figure(figsize=(10, 8))
        for t in tlist:
            u = self.Q[t][1] / self.Q[t][0]
            plt.plot([self.delta_x * i for i in range(self.num_x)], u, c=cm.winter(1 - t/tlist[-1]), label='t = {}'.format(t/1000000))
        plt.legend()
        plt.savefig('../output/prob1/u.png', dpi=200, bbox_inches='tight')

        fig = plt.figure(figsize=(10, 10))
        t = tlist[-1]
        rho = self.Q[t][0]
        p = self.E[t][1] - self.Q[t][1] * self.Q[t][1] / self.Q[t][0]
        u = self.Q[t][1] / self.Q[t][0]
        plt.subplot(3, 1, 1)
        plt.plot([self.delta_x * i for i in range(self.num_x)], rho, c='b', label='ρ')
        plt.legend()
        plt.subplot(3, 1, 2)
        plt.plot([self.delta_x * i for i in range(self.num_x)], p, c='r', label='p')
        plt.legend()
        plt.subplot(3, 1, 3)
        plt.plot([self.delta_x * i for i in range(self.num_x)], u, c='limegreen', label='u')
        plt.legend()
        plt.savefig('../output/prob1/compare.png', dpi=200, bbox_inches='tight')

if __name__ == "__main__":
    start = time.time()
    mc = MacCormack(1, 0.2, 0.001, 0.000001)
    left_init = [1, 1, 0]
    right_init = [0.1, 0.125, 0]
    mc.set_init(left_init, right_init)
    mc.simulate()
    end = time.time()
    print('実行時間 : {:.2f}s'.format(end - start))
    mc.visualize(np.array([0, 0.05, 0.1, 0.15, 0.2]))
