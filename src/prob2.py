import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
sns.set()
import time


class AUSM:
    def __init__(self, xrange, trange, delta_x, delta_t):
        self.num_x = int(xrange / delta_x) + 1
        self.num_t = int(trange / delta_t) + 1
        self.Q = np.zeros((self.num_t, 3, self.num_x))
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
    

    def get_M12(self, Q):
        u = Q[1] / Q[0]
        p = (self.gamma - 1) * (Q[2] - 0.5 * Q[1] * Q[1] / (Q[0]+1e-9))
        a = (self.gamma * p / Q[0])**0.5
        M = u / a
        
        Mp = np.zeros(M.shape)
        Mm = np.zeros(M.shape)
        p_idx = abs(M) > 1
        m_idx = abs(M) <= 1

        Mp[p_idx] = 0.5 * (M[p_idx] + abs(M[p_idx]))
        Mm[p_idx] = 0.5 * (M[p_idx] - abs(M[p_idx]))
        
        Mp[m_idx] = + 0.25 * (M[m_idx] + 1)**2
        Mm[m_idx] = - 0.25 * (M[m_idx] - 1)**2
        return a, p, M, Mp[:-1] + Mm[1:]

    
    def get_p12(self, p, M):
        pp = np.zeros(p.shape)
        pm = np.zeros(p.shape)
        p_idx = abs(M) > 1
        m_idx = abs(M) <= 1

        pp[p_idx] = 0.5 * p[p_idx] * (M[p_idx] + abs(M[p_idx])) / M[p_idx]
        pm[p_idx] = 0.5 * p[p_idx] * (M[p_idx] - abs(M[p_idx])) / M[p_idx]
        
        pp[m_idx] = 0.5 * p[m_idx] * (1 + M[m_idx])
        pm[m_idx] = 0.5 * p[m_idx] * (1 - M[m_idx])
        return pp[:-1] + pm[1:]

    

    def get_E12(self, Q):
        a, p, M, M12 = self.get_M12(Q)
        p12 = self.get_p12(p, M)

        Q[2] += p
        p_idx = M12 >= 0
        m_idx = M12 < 0

        Qa = Q * a
        E12 = np.zeros((3, M12.shape[0]))
        E12[:, p_idx] = M12[p_idx] * Qa[:, :-1][:, p_idx]
        E12[:, m_idx] = M12[m_idx] * Qa[:, 1:][:, m_idx]
        E12[1] += p12
        return E12
    

    def update(self):
        E12 = self.get_E12(self.Q[self.cnt].copy())
        self.Q[self.cnt+1][:, 1:-1] = self.Q[self.cnt][:, 1:-1] - self.delta_t / self.delta_x * (E12[:, 1:] - E12[:, :-1])
        self.Q[self.cnt+1][:, 0] = self.Q[self.cnt][:, 0]
        self.Q[self.cnt+1][:, -1] = self.Q[self.cnt][:, -1]
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
        plt.savefig('../output/prob2/rho.png', dpi=200, bbox_inches='tight')
        
        # p
        fig = plt.figure(figsize=(10, 8))
        for t in tlist:
            p = (self.gamma - 1) * (self.Q[t][2] - 0.5 * self.Q[t][1] * self.Q[t][1] / (self.Q[t][0]+1e-9))
            plt.plot([self.delta_x * i for i in range(self.num_x)], p, c=cm.winter(1 - t/tlist[-1]), label='t = {}'.format(t/1000000))
        plt.legend()
        plt.savefig('../output/prob2/p.png', dpi=200, bbox_inches='tight')

        # u
        fig = plt.figure(figsize=(10, 8))
        for t in tlist:
            u = self.Q[t][1] / self.Q[t][0]
            plt.plot([self.delta_x * i for i in range(self.num_x)], u, c=cm.winter(1 - t/tlist[-1]), label='t = {}'.format(t/1000000))
        plt.legend()
        plt.savefig('../output/prob2/u.png', dpi=200, bbox_inches='tight')


if __name__ == "__main__":
    start = time.time()
    ausm = AUSM(1, 0.2, 0.001, 0.000001)
    left_init = [1, 1, 0]
    right_init = [0.1, 0.125, 0]
    ausm.set_init(left_init, right_init)
    ausm.simulate()
    end = time.time()
    print('実行時間 : {:.2f}s'.format(end - start))
    ausm.visualize(np.array([0, 0.05, 0.1, 0.15, 0.2]))