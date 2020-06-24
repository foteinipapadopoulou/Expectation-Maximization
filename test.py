import matplotlib.pyplot as plt
import matplotlib.image as img
import numpy as np
K = 1
cost = np.load("costs_k_"+str(K)+".npy")
plt.plot(cost)
plt.ylabel('log likelihood')
plt.xlabel('iterations ')
plt.title("Log likelihood ")
plt.show()
gamma = np.load("gamma_file_k_"+str(K)+".npy")
mean = np.load("mean_file_k_"+str(K)+".npy")
rec_img = np.zeros((gamma.shape[0],mean.shape[1]))
for n in range(gamma.shape[0]):
    rec_img[n] = mean[gamma[n].argmax()]
image=img.imread('im.jpg')
plt.imshow(image)
plt.show()
array=np.array(image)
array.shape

new_img = array.reshape((array.shape[0]*array.shape[1]), array.shape[2])
new_img = new_img/255

plt.imshow(np.reshape(rec_img,(array.shape[0],array.shape[1],array.shape[2])))
plt.show()
from numpy import linalg as LA
N=rec_img.shape[0]
error=np.square(LA.norm(new_img-rec_img))/N
print(error)