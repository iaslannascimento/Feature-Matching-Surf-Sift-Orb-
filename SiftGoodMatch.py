#Autor: Iaslan Nascimento
#25/11/19
#código para detecção de memes utilizando a abordagem SIFT para matching de pontos
#adição dia 26/11/19
#Porcentagem de matching e verificação de boas caracteristicas entre imagens.
import numpy as np
import cv2 
import matplotlib.pyplot as plt

im1 = cv2.imread('009.jpg')
im2 = cv2.imread('005.jpg')

#inicializando o sift 
sift = cv2.xfeatures2d.SIFT_create()

#encontrando pontos chave e descrições com sift
kp1,des1 = sift.detectAndCompute(im1,None)
kp2,des2 = sift.detectAndCompute(im2,None)

#chamando as funções FLANN que possuem diversos algoritmos de optimização para os vizinhos mais próximos
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params,search_params)

matches = flann.knnMatch(des1,des2,k=2)
matchesMask = [[0,0] for i in range(len(matches))]

#quantidade minima de matches necessária para uma boa confirmação
qtdMinMatches = 50
#vetor para armazenar apenas as boas caracteristicas
good = []
#pontos da imagem base 
pontosBase = len(kp1)
for i,(m,n) in enumerate(matches):
    if m.distance < 0.7*n.distance:
        matchesMask[i]=[1,0]
        good.append(m) 
        
if len(good) > qtdMinMatches:
    
    bons = len(good)
    resultado = bons * 100/pontosBase
    print(resultado)
    
    
draw_params = dict(matchColor = (255,255,0), singlePointColor = (0,0,0), matchesMask = matchesMask, flags = cv2.DrawMatchesFlags_DEFAULT)
im3 = cv2.drawMatchesKnn(im1,kp1,im2,kp2,matches,None,**draw_params)
plt.imshow(im3),plt.show()
