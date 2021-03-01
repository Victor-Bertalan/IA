import numpy as np
from skimage import io
imgs=np.zeros((9,400,600))
for i in range(9):
    img=np.load("images/car_"+str(i)+".npy")
    imgs[i]=img

print(np.sum(imgs))
sume=np.sum(imgs, axis=(1,2))
print(sume)
print(np.argmax(sume,0))
dev=np.std(imgs,axis=(0))
print(dev)
print(np.shape(dev))

mean_image=np.mean(imgs,0)
io.imshow(mean_image.astype(np.uint8))
io.show()

imgnorm=np.zeros((9,400,600))
for i in range(9):
    imgnorm[i]=(imgs[i]-mean_image)/dev[i]
    io.imshow(imgnorm[i].astype(np.uint8))
    io.show()

print(imgnorm)

slice=imgs[:,200:300,280:400]
for i in range(9):
    io.imshow(slice[i].astype(np.uint8))
    io.show()
