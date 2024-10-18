#!/usr/bin/python

import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots()

xdata = np.array([8, 9, 6, 1, 4, 2])
ydata = np.array([7, 9, 8, 3, 3, 2])

xdata = xdata - xdata.mean()
ydata = ydata - ydata.mean()

ax.spines["right"].set_color("none")
ax.spines["top"].set_color("none")

ax.spines["left"].set_position("center")
ax.spines["bottom"].set_position("center")

datmat = np.array([xdata, ydata])

cov = np.cov(xdata, y=ydata)
#same calculation
#cov = np.array((np.matrix(datmat) * np.matrix(datmat).T) / (len(xdata) - 1))

eigvals, eigvecs = np.linalg.eig(cov)
origin = np.array([[0,0], [0,0]])
print("eig")
print(eigvecs)
print("eig")

# eigvecs are stored in columns, we need them in rows
eigvecs = eigvecs.T

for vec in eigvecs:
    ax.axline((0, 0), xy2=vec, c ='r')

mat = np.array([xdata, ydata])
res = np.matmul(eigvecs, mat)

#print("transformed:")
#print(mat.transpose() * eigvecs.transpose())

print()
print("res:")
print(np.cov(res[0], y=res[1]))

ax.scatter(res[0], res[1], c='r')
ax.scatter(xdata, ydata)
plt.show()

