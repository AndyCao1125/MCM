import matplotlib.pyplot as plt
import math
import random
import numpy as np
import matplotlib.pyplot as plt
T_init = 2  # 初始最大温度
alpha = 0.95  # 降温系数
T_min = 1e-3  # 最小温度，即退出循环条件

def obj(x):
    y = np.sin(10 * np.pi * x) / x
    return -y


def SA(T_init, alpha, T_min):
    T = T_init
    x_new = random.random() * 10  # 初解
    x_current = x_new
    y_current = float('inf')
    x_best = x_new
    y_best = float('inf')
    fitness=[]
    while T > T_min:
        for i in range(100):
            delta_x = random.random() - 0.5
            # 自变量变化后仍要求在[0,10]之间
            if 0 < (x_new + delta_x) < 10:
                x_new = x_new + delta_x
            else:
                x_new = x_new - delta_x
            y_new = obj(x_new)

            if (y_new < y_current):
                y_current = y_new
                x_current = x_new
                if (y_new < y_best):
                    y_best = y_new
                    x_best = x_new
                    fitness.append(obj(x_best))
                    #print(x_best)       #想把每步的xbest提取出来 然后画图
            else:
                if random.random() < math.exp(-(y_new - y_current) / T):
                    y_current = y_new
                    x_current = x_new
                else:
                    x_new = x_current
        #fitness.append(x_best)
        T *= alpha
    print('最优解', x_best, obj(x_best))
    return fitness

fitness=SA(T_init, alpha, T_min)
#SA(T_init, alpha, T_min)
plt.figure()
plt.title("Figure SA")
plt.xlabel("iterators", size=14)
plt.ylabel("fitness", size=14)
t = np.linspace(0, 100, len(fitness))
fitness = np.array(fitness)
plt.plot(t, fitness, color='b', linewidth=3)
plt.show()

