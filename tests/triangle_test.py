import numpy as np
import matplotlib.pyplot as plt

vertices = [(0,0), (8,0), (0,2)]

# draw the triangle
plt.plot([0, 8], [0, 0])
plt.plot([0, 0], [0, 2])
plt.plot([0, 8], [2, 0])

# take the mean of the three points
vertices = np.stack(vertices).astype(float)
mean_pos = vertices.mean(axis=0)

# draw the mean as the point on the plot
plt.scatter(mean_pos[0], mean_pos[1])
plt.show()

# subtract the mean from the vertices and redraw the triangle
vertices -= mean_pos[None]
plt.plot((vertices[0,0], vertices[1,0]), (vertices[0,1], vertices[1,1]))
plt.plot((vertices[0,0], vertices[2,0]), (vertices[0,1], vertices[2,1]))
plt.plot((vertices[2,0], vertices[1,0]), (vertices[2,1], vertices[1,1]))
zero_mean = vertices.mean(axis=0)

plt.scatter(zero_mean[0], zero_mean[1])
plt.show()

# compute the bounding box
min_v = vertices.min(axis=0)
max_v = vertices.max(axis=0)
print(np.absolute(min_v) - np.absolute(max_v))

# now I will do the same thing for equilateral triangle
vertices = [(2,1), (7,1), (4.5, 5.33)]
vertices = np.stack(vertices)
mean_pos = vertices.mean(axis=0)
print(mean_pos)
plt.plot((vertices[0,0], vertices[1,0]), (vertices[0,1], vertices[1,1]))
plt.plot((vertices[0,0], vertices[2,0]), (vertices[0,1], vertices[2,1]))
plt.plot((vertices[2,0], vertices[1,0]), (vertices[2,1], vertices[1,1]))
plt.scatter(mean_pos[0], mean_pos[1])
plt.show()

vertices = vertices - mean_pos[None]
plt.plot((vertices[0,0], vertices[1,0]), (vertices[0,1], vertices[1,1]))
plt.plot((vertices[0,0], vertices[2,0]), (vertices[0,1], vertices[2,1]))
plt.plot((vertices[2,0], vertices[1,0]), (vertices[2,1], vertices[1,1]))
new_mean_pos = vertices.mean(axis=0)
print(new_mean_pos)
plt.scatter(new_mean_pos[0], new_mean_pos[1])
plt.show()

# now compute the max and min and subtraction would only be zero in x
max_v = np.max(vertices, axis=0)
min_v = np.min(vertices, axis=0)
print(np.absolute(min_v) - np.absolute(max_v))

