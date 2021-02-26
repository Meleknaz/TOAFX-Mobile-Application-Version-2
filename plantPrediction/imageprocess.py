import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imshow
from skimage.filters import prewitt_h,prewitt_v


## Images will be upl. from cloud(pt) or local files.
img = cv2.imread('C:/Users/Asus/Desktop/TOAFX/resim.jpeg')

##Gaussian Blur
# img = np.copy(img)
# for i in range(1,31,2):
# 	img=cv2.GaussianBlur(img,(i,i),0)


scale_percent = 60
width = int(img.shape[1]*scale_percent/100)
height = int(img.shape[0]*scale_percent/100)
dim = (width,height)
resized = cv2.resize(img,dim,interpolation = cv2.INTER_AREA)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
pixel_values = img.reshape((-1,3))
pixel_values = np.float32(pixel_values)

# print(pixel_values.shape)

criter = (cv2.TERM_CRITERIA_EPS + cv2.TermCriteria_MAX_ITER,100,0.0001)
# #number of cluster (green for plant , yellow for disease , brown for others)
k = 3
flags = cv2.KMEANS_RANDOM_CENTERS

compactness, labels, (centers) = cv2.kmeans(pixel_values,k,None,criter,50,flags)
centers = np.uint8(centers)
labels = labels.flatten()
segmented_image = centers[labels.flatten()]
segmented_image = segmented_image.reshape(img.shape)
removedCluster = 1
cannyImage = np.copy(segmented_image).reshape((-1,3))
cannyImage[labels.flatten() == removedCluster] = [0, 0, 0]
cannyImage = cv2.Canny(cannyImage,25,200).reshape(segmented_image.shape)

print(compactness)



####GAUSSION BLURING PRE-PROCESSING
# dst = np.copy(cannyImage)

# for i in range(1,31,2):
# 	dst=cv2.GaussianBlur(segmented_image,(i,i),0)
# plt.imshow(dst)
# plt.show()

# cv2.imwrite('C:/Users/Asus/Desktop/TOAFX/res.jpeg',cannyImage)

cv2.imshow("Lol",cannyImage)
    


cv2.waitKey(0)
cv2.destroyAllWindows()
