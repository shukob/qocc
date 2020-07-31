import matplotlib.pyplot as plt
import numpy as np


def foo(x,y):
    '''

    :param x:
    :param y:
    :return:
    '''
    return x+y
''' gerar figura 1 '''
fig, ax = plt.subplots()
ax.axis("equal")
circle = plt.Circle((0,0), 1, edgecolor='black', facecolor='None')
ax.add_artist(circle)


# pontos do conjunto de treinamento
points = []

for k in range(10):
    angle = (0.7 + k/30) *np.pi
    point = [np.cos(angle), np.sin(angle)]
    points.append(np.array(point))
    # print(0.7 + k/30)
    ax.plot(point[0], point[1], '*', color ='b')

# ponto de teste x_0
angle = ((0.7 + 1)/2) * np.pi  + np.pi
#point = [np.cos(angle), np.sin(angle)]
point = points[0] + points[9]
point = -(point / np.linalg.norm(point))
x0 = np.array(point)
ax.plot(point[0], point[1], 's', color ='black')
ax.annotate('$x_0$',point, (point[0]-0.2, point[1]+0.1))

# ponto de teste x_1
angle = ((0.7 + 1)/2) * np.pi
point = -x0
x1 = np.array(point)
ax.plot(point[0], point[1], 's', color ='black')
ax.annotate('$x_1$',point, (point[0]+0.1, point[1]-0.1))

ax.set_xlim((-1.5, 1.5))
ax.set_ylim((-1.5, 1.5))
ax.grid(fillstyle='full')
fig.savefig('fig1.pdf')

'''pacc x0 e x1'''

# pacc x0
pacc = 0
for point in points:
    pacc += np.linalg.norm(point + x0)**2
pacc = pacc/(4 * len(points))
print('pacc x0 =', pacc)

# pacc x1
pacc = 0
for point in points:
    pacc += np.linalg.norm(point + x1)**2
pacc = pacc/(4 * len(points))
print('pacc x1 =', pacc)