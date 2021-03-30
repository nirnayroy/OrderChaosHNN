from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt

def z1(x, y):
	return -((
		(1/(1+(np.exp(-(x+2)/0.01))))-(1/(1+(np.exp(-(x-2)/0.01)))))*(
		(1/(1+(np.exp(-(y+2)/0.01))))-(1/(1+(np.exp(-(y-2)/0.01))))))
x = np.outer(np.linspace(-3, 3, 30), np.ones(30))
y = x.copy().T # transpose
z = z1(x, y)

fig = plt.figure()
ax = plt.axes(projection='3d')

ax.plot_surface(x, y, z,cmap='viridis', edgecolor='none')
ax.set_title('Surface plot')
plt.savefig('potplot.png')