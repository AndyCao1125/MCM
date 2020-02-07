import numpy as np
import random
import matplotlib.pyplot as plt


# ----------------------PSO参数设置---------------------------------
class PSO():

    def __init__(self, pN, dim, max_iter):  # 初始化类  设置粒子数量  位置信息维度  最大迭代次数
        # self.w = 0.8
        self.ws = 0.9
        self.we = 0.4
        self.c1 = 1.49445
        self.c2 = 1.49445
        self.r1 = 0.6
        self.r2 = 0.3
        self.pN = pN  # 粒子数量
        self.dim = dim  # 搜索维度
        self.max_iter = max_iter  # 迭代次数
        self.X = np.zeros((self.pN, self.dim))  # 所有粒子的位置（还要确定取值范围）
        self.Xmax = 2
        self.Xmin = 1
        self.V = np.zeros((self.pN, self.dim))  # 所有粒子的速度（还要确定取值范围）
        self.Vmax = 0.5
        self.Vmin = -0.5
        self.pbest = np.zeros((self.pN, self.dim))  # 个体经历的最佳位置
        self.gbest = np.zeros((1, self.dim))  # 全局最佳位置
        self.p_fit = np.zeros(self.pN)  # 每个个体的历史最佳适应值
        self.fit = 0  # 全局最佳适应值

    # ---------------------目标函数Sphere函数-----------------------------
    def function(self, x):
        y = np.sin(10 * np.pi * x) / x
        return y

    # ---------------------初始化种群----------------------------------
    def init_Population(self):
        for i in range(self.pN):  # 遍历所有粒子

            for j in range(self.dim):  # 每一个粒子的纬度
                self.X[i][j] = random.uniform(1, 2)  # 给每一个粒子的位置赋一个初始随机值（在一定范围内）
                self.V[i][j] = random.uniform(-0.5, 0.5)  # 给每一个粒子的速度给一个初始随机值（在一定范围内）

            self.pbest[i] = self.X[i]  # 把当前粒子位置作为这个粒子的最优位置
            tmp = self.function(self.X[i])  # 计算这个粒子的适应度值
            self.p_fit[i] = tmp  # 当前粒子的适应度值作为个体最优值

            if (tmp > self.fit):  # 与当前全局最优值做比较并选取更佳的全局最优值
                self.fit = tmp
                self.gbest = self.X[i]

            # ---------------------更新粒子位置----------------------------------

    def iterator(self):
        fitness = []
        for t in range(self.max_iter):
            w = self.ws - (self.ws - self.we) * (t / self.max_iter)
            for i in range(self.pN):

                # 更新速度
                self.V[i] = w * self.V[i] + self.c1 * self.r1 * (self.pbest[i] - self.X[i]) + self.c2 * self.r2 * (
                            self.gbest - self.X[i])
                if self.V[i] > self.Vmax:
                    self.V[i] = self.Vmax
                elif self.V[i] < self.Vmin:
                    self.V[i] = self.Vmin

                # 更新位置
                self.X[i] = self.X[i] + self.V[i]
                if self.X[i] > self.Xmax:
                    self.X[i] = self.Xmax
                elif self.X[i] < self.Xmin:
                    self.X[i] = self.Xmin

            for i in range(self.pN):  # 更新gbest\pbest

                temp = self.function(self.X[i])

                if (temp > self.p_fit[i]):  # 更新个体最优
                    self.pbest[i] = self.X[i]
                    self.p_fit[i] = temp

                if (temp > self.fit):  # 更新全局最优
                    self.gbest = self.X[i]
                    self.fit = temp

            fitness.append(self.fit)
            print('最优值为：', self.fit)  # 输出最优值
            x1 = self.fit
            print('最优位置为：', self.X[i])
            y1 = self.X[i]
        return fitness, x1, y1


# ----------------------程序执行-----------------------
my_pso = PSO(pN=20, dim=1, max_iter=50)
my_pso.init_Population()
fitness, x1, y1 = my_pso.iterator()

plt.figure(1)
plt.title("Figure1")
plt.xlabel("iterators", size=14)
plt.ylabel("fitness", size=14)
t = np.array([t for t in range(0, 50)])
fitness = np.array(fitness)
plt.plot(t, fitness, color='b', linewidth=3)
plt.show()
#aa = np.arange(0,10,0.2)
plt.figure()
#plt.plot(aa, np.sin(10 * np.pi * aa) / aa, color='b', linewidth=3)
plt.scatter(y1, x1, marker='x', color='red', s=100, label='First')
plt.show()
