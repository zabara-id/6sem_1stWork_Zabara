import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# инициализация параметров
T = 18
h = 0.5
L = 20 - h
CFL = 0.1
tau = h * CFL
N = int(L / h)
x = np.linspace(0, L, N + 1)
y = np.zeros((int(T / tau), N + 1))
y0 = np.sin(4 * np.pi * x / (L + h))
y[0] = y0
y_next = y0

# реализация схемы "уголок"
for i in range(int(T / tau) - 1):
    for j in range(N):
        y_next[j + 1] = y[i][j + 1] * (1 - CFL) + CFL * y[i][j]
    y_next[0] = y[i][0] * (1 - CFL) + CFL * y[i][-1]
    y[i + 1] = y_next

# визуализация
fig = plt.figure()
ax = plt.axes(xlim=(0, 20), ylim=(-2, 2))
line, = ax.plot([], [], lw=3)


def init():
    line.set_data([], [])
    return line,


def animate(i):
    global y
    x = np.linspace(0, L, N + 1)
    y_anim = y[i]
    line.set_data(x, y_anim)
    return line,


anim = FuncAnimation(fig, animate, init_func=init,
                     frames=int(T / tau), interval=40, blit=True)

anim.save('sine_wave_corner.gif', writer='pillow')

plt.show()
