import cv2
from matplotlib import pyplot as plt
im1 = cv2.imread('im5.jpg')#,cv2.COLOR_BGR2GRAY)
im2 = cv2.imread('im6.jpg')#,cv2.COLOR_BGR2GRAY)

plt.imshow(im1)
plt.show()

plt.imshow(im2)
plt.show()
#iniciando o detector sift 
orb = cv2.ORB_create()
kp1,des1 = orb.detectAndCompute(im1,None)
kp2,des2 = orb.detectAndCompute(im2,None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1,des2)
matches =sorted(matches, key = lambda x:x.distance)
im3 = cv2.drawMatches(im1,kp1,im2,kp2,matches[:10],None, flags=2)
plt.imshow(im3),plt.show()
