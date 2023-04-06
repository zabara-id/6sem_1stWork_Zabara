import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

L = 10
T = 0.02
h = 0.05
CFL = 0.001
tau = h * CFL
N = int(2*L/h)
x = np.linspace(-L, L, N+1)
gamma = 5/3
ul = 0
ql = 13
Pl = 10 * 101325
ur = 0
qr = 1.3
Pr = 1 * 101325
wol = [ql, ql * ul, Pl/(gamma - 1)]
wor = [qr, qr * ur, Pr/(gamma - 1)]

y0 = []
for i in range(N+1):
    if x[i] < 0:
        y0.append(wol)
    else:
        y0.append(wor)

y = [y0]


def for_calculation(w, gamma):
    e = w[2]/w[0]
    u = w[1]/w[0]
    c = np.sqrt(gamma * (gamma - 1) * e)
    matrix_lambda = np.array([[u+c, 0, 0], [0, u, 0], [0, 0, u - c]])
    mod_matrix_lambda = np.abs(matrix_lambda)
    owega = np.array([[-u*c, c, gamma - 1], [-c*c, 0, gamma - 1], [u*c, -c, gamma - 1]])
    inv_owega = np.linalg.inv(owega)
    A = np.array([[0, 1, 0], [-u*u, 2*u, gamma-1], [-gamma*u*e, gamma*e, u]])
    mod_A = np.matmul(np.matmul(inv_owega, mod_matrix_lambda), owega)

    return A, mod_A

y_now = y0.copy()
y_next = y0.copy()
print(y[0])
for i in range(int(T/tau)):
    for j in range(N-1):
        w_n_l_minus1 = np.array([[y_now[j + 0][0]], [y_now[j + 0][1]], [y_now[j + 0][2]]])
        w_n_l = np.array([[y_now[j + 1][0]], [y_now[j + 1][1]], [y_now[j + 1][2]]])
        w_n_l_plus1 = np.array([[y_now[j + 2][0]], [y_now[j + 2][1]], [y_now[j + 2][2]]])
        A, mod_A = for_calculation(y_now[j+1], gamma)
        w_n_plus1 = w_n_l - CFL/2*np.matmul(A, (w_n_l_plus1 - w_n_l_minus1)) + CFL/2*np.matmul(mod_A, w_n_l_plus1 - 2*w_n_l
                                                                                               + w_n_l_minus1)
        y_next[j+1] = [w_n_plus1[0][0], w_n_plus1[1][0], w_n_plus1[2][0]]

    y_next[0] = y_next[1]
    y_next[-1] = y_next[-2]
    y_now = y_next.copy()
    y.append(y_now)
    print(i/4, "%")


fig = plt.figure()
ax0 = fig.add_subplot(221, xlim=(-L, L), ylim=(0, 15))
ax0.set_title("Плотность кг/м^3")
line1, = ax0.plot([], [], lw=2)
ax1 = fig.add_subplot(222, xlim=(-L, L), ylim=(-40, 400))
ax1.set_title("Скорость м/с")
line2, = ax1.plot([], [], lw=2, color='r')
ax2 = fig.add_subplot(223, xlim=(-L, L), ylim=(60, 200))
ax2.set_title("Энергия кДж/кг")
line3, = ax2.plot([], [], lw=2, color='y')
ax3 = fig.add_subplot(224, xlim=(-L, L), ylim=(0, 11))
ax3.set_title("Давление атм")
line4, = ax3.plot([], [], lw=2, color='g')
line = [line1, line2, line3, line4]

def init():
    line[0].set_data([], [])
    line[1].set_data([], [])
    line[2].set_data([], [])
    line[3].set_data([], [])
    return line


def animate(i):
    global y
    x = np.linspace(-L, L, N+1)
    y_anim_q = []
    y_anim_u = []
    y_anim_e = []
    y_anim_P = []
    for a in range(N+1):
        y_anim_q.append(y[i][a][0])
        y_anim_u.append(y[i][a][1]/y[i][a][0])
        y_anim_e.append(y[i][a][2] / y[i][a][0]/ 1000)
        y_anim_P.append(y[i][a][2] * (gamma - 1)/ 101325)
    line[0].set_data(x, y_anim_q)
    line[1].set_data(x, y_anim_u)
    line[2].set_data(x, y_anim_e)
    line[3].set_data(x, y_anim_P)
    return line



anim = FuncAnimation(fig, animate, init_func=init,
                     frames=int(T/tau), interval=30, blit=True)

anim.save('sine_wave_kir.gif', writer='pillow')


plt.show()