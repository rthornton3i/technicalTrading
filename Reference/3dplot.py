import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax = plt.gca()
ax = plt.axes(projection='3d')

scatter = ax.scatter3D(x, y, z, c=v, s=np.asarray(d)*5, cmap='viridis')
scatter = ax.scatter3D(x, y, z, c=v, cmap='viridis')

ax.set_xlabel('fast')
ax.set_ylabel('slow')
ax.set_zlabel('value')
cb = fig.colorbar(scatter,ax=ax)
cb.set_label('signal')