import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_excel('MACD_test.xlsx',index_col=0)

df2 = df[df['f']>35500]

x = df2['x']
y = df2['y']
z = df2['f']
col = df2['z']

fig = plt.figure()
ax = plt.axes(projection='3d')

# ax.plot_surface(X, Y, Z, rstride=1, cstride=1,cmap='viridis', edgecolor='none')
scatter = ax.scatter3D(x, y, z, c=col, cmap='viridis')

ax.set_xlabel('fast')
ax.set_ylabel('slow')
ax.set_zlabel('value')
cb = fig.colorbar(scatter,ax=ax)
cb.set_label('signal')