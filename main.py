import numpy as np
import matplotlib.pyplot as plt

x = np.array([9, 10], dtype=float)
eps = 0.01


def system(x):
    f1 = x[0] - np.cos(x[1]) - 10
    f2 = x[1] - np.sin(x[0]) - 10
    return np.array([f1, f2], dtype=float)


def simple_iteration(x0, eps):
    x = x0.copy()
    while True:
        x1_next = 10 + np.cos(x[1])
        x2_next = 10 + np.sin(x[0])
        x_next = np.array([x1_next, x2_next])
        if np.linalg.norm(x_next - x) < eps:
            break
        x = x_next
    return x


def jacobi(x):
    j11 = 1
    j12 = np.sin(x[1])
    j21 = -np.cos(x[0])
    j22 = 1
    return np.array([[j11, j12],
                     [j21, j22]], dtype=float)


def newton(x0, eps):
    x = x0.copy()
    while True:
        dx = np.linalg.solve(jacobi(x), -system(x))
        x += dx
        if np.linalg.norm(dx) < eps:
            break
    return x

def plot():
    x=np.linspace(0,12)
    x1=np.cos(x)+10
    x2=np.sin(x)+10
    plt.plot(x1,x,x,x2)
    plt.show()
    plt.close()


print("Метод простих ітерацій:", simple_iteration(x, eps))
print("Метод Ньютона:", newton(x, eps))
print(plot())
