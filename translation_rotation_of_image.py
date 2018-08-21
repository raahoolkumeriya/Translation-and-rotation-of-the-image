import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

# Function for rotating image
def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
 
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
 
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
 
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
 
    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))


#Defining size for the input iamge

size=(1024,1024)

original_img = Image.open('t-img.tif')

resized_image = original_img.resize(size)

resized_image.save('resize_img.tif')

img = cv2.imread('resize_img.tif',0)
rows,cols = img.shape


fig, axarr = plt.subplots(2, 2)
fig.suptitle("Image Translation", fontsize=16)

res = cv2.resize(img,None,fx=2, fy=2, interpolation = cv2.INTER_CUBIC)

## M = np.float32([[1,0,X],[0,1,Y]])

letf_translation = np.float32([[1,0,50],[0,1,0]])
leftT = cv2.warpAffine(img,letf_translation,(cols,rows))

#Letf translation and Rotation
#rotate_bound(leftT,30)


right_translation = np.float32([[1,0,-50],[0,1,0]])
rightT = cv2.warpAffine(img,right_translation,(cols,rows))

bottom_translation = np.float32([[1,0,1],[0,1,50]])
bottomT = cv2.warpAffine(img,bottom_translation,(cols,rows))

up_translation = np.float32([[1,0,1],[0,1,-50]])
upT = cv2.warpAffine(img,up_translation,(cols,rows))



plt.subplot(231),plt.imshow(img),plt.title('Original Image')
plt.subplot(232),plt.imshow(resized_image),plt.title('Resize Image 1024x1024')
plt.subplot(233),plt.imshow(leftT),plt.title('Left Translated Image')
plt.subplot(234),plt.imshow(rightT),plt.title('Right Translated Image')
plt.subplot(235),plt.imshow(bottomT),plt.title('Bottom Translated Image')
plt.subplot(236),plt.imshow(upT),plt.title('Up Translated Image')

plt.figure()
#"Left Translation and Rotation", fontsize=18)
plt.subplot(121),plt.imshow(leftT),plt.title('Left Translated Image')
plt.subplot(122),plt.imshow(rotate_bound(leftT,30)),plt.title('Rotated Image')



plt.figure()
# Right Translation and Rotation", fontsize=18)
plt.subplot(121),plt.imshow(rightT),plt.title('Right Translated Image')
plt.subplot(122),plt.imshow(rotate_bound(rightT,30)),plt.title('Rotated Image')

plt.figure()
# Bottom Translation and Rotation", fontsize=18)
plt.subplot(121),plt.imshow(bottomT),plt.title('Bottom Translated Image')
plt.subplot(122),plt.imshow(rotate_bound(bottomT,30)),plt.title('Rotated Image')

plt.figure()
# UP Translation and Rotation", fontsize=18)
plt.subplot(121),plt.imshow(upT),plt.title('Up Translated Image')
plt.subplot(122),plt.imshow(rotate_bound(upT,30)),plt.title('Rotated Image')


fig.tight_layout()
fig.subplots_adjust(top=0.88)

plt.show()


