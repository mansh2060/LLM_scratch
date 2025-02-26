import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

inputs=torch.tensor(
    [[0.43,0.15,0.89], # your
     [0.55,0.87,0.66], # journey
     [0.57,0.85,0.64], # starts
     [0.22,0.58,0.33], # with
     [0.77,0.25,0.10], # one
     [0.05,0.80,0.55]  # step
     ]
)
tokens=['Your','Journey','Starts','With','One','Step']
x_cordinates=inputs[:,0].numpy()
y_cordinates=inputs[:,1].numpy()
z_cordinates=inputs[:,2].numpy()
print(x_cordinates)
print(y_cordinates)
print(z_cordinates)

fig=plt.figure()
axis=fig.add_subplot(111,projection='3d')

for x,y,z,word in zip(x_cordinates,y_cordinates,z_cordinates,tokens):
    axis.scatter(x,y,z)
    axis.text(x,y,z,word,fontsize=10)

axis.set_xlabel('X')
axis.set_ylabel('Y')
axis.set_zlabel('Z')

plt.title('3D Plot of Word Embeddings')
plt.show()

fig=plt.figure()
axis=fig.add_subplot(111,projection='3d')

colors = ['r', 'g', 'b', 'c', 'm', 'y']
for (x, y, z, word, color) in zip(x_cordinates, y_cordinates, z_cordinates, tokens, colors):
    axis.quiver(0, 0, 0, x, y, z, color=color, arrow_length_ratio=0.05)
    axis.text(x, y, z, word, fontsize=10, color=color)

axis.set_xlabel('X')
axis.set_ylabel('Y')
axis.set_zlabel('Z')

axis.set_xlim([0, 1])
axis.set_ylim([0, 1])
axis.set_zlim([0, 1])

plt.title('3D Plot of Word Embeddings with Colored Vectors')
plt.show()
