# Assignment 5 for 6.815/865
# Submission:
# Deadline: Oct 9
# Your name: Ryan Lacey
# Reminder: 
# - Don't hand in data
# - Don't forget README.txt

import numpy as np
from bilagrid import bilateral_grid

##def BW(im, weights=[0.3,0.6,0.1]):
##    return np.dot(im.copy(), np.array([weights]).T)

def BW(im, weights=[0.4,0.7,0.01]):
   img = im.copy()
   (height, width, rgb) = np.shape(img)
   for y in xrange(height):
       for x in xrange(width):
           img[y][x] = np.dot(img[y][x], weights)
   return img


def lumiChromi(im):
    imL = im.copy()
    imC = im.copy()
    imL = BW(imL)
    imC = im / imL
    return (imL, imC)


def computeWeight(im, epsilonMini=0.002, epsilonMaxi=0.99):
    out = np.ones(np.shape(im))
    out[im < epsilonMini] = 0
    out[im > epsilonMaxi] = 0
    return out 

def computeFactor(im1, w1, im2, w2):
    scaled = im2/ (im1 + 0.0000000001)
    scaled[w1 == 0] = 0
    scaled[w2 == 0] = 0
    nonzeroes = scaled[np.nonzero(scaled)]
    sort = np.sort(nonzeroes)
    return sort[len(sort)/2]    

def makeHDR(imageList, epsilonMini=0.002, epsilonMaxi=0.99):      
    imgWeights = [computeWeight(img, epsilonMini, epsilonMaxi) for img in imageList]
    imgWeights[0] = computeWeight(imageList[0], epsilonMini, 1)
    imgWeights[-1] = computeWeight(imageList[-1], 0, epsilonMaxi)

    out = imageList[0] * imgWeights[0]    
    weightSum = imgWeights[0]
    factor = 1    
    for i in range(1, len(imageList)):
        weightSum += imgWeights[i]
        factor = computeFactor(imageList[i-1], imgWeights[i-1], imageList[i], imgWeights[i]) * factor
        out += (imgWeights[i] * (1/factor) * imageList[i])
        
    out /= weightSum
    return out
    
def toneMap(im, targetBase=100, detailAmp=1, useBila=False):
    (lum, chrom) = lumiChromi(im)
    # find smallest non-zero value
    smallestValue = np.min(lum[np.nonzero(lum)])
    # set zero values to smallest value
    lum[lum == 0] = smallestValue
    # change lum to log domain
    lumLogged = np.log10(lum)
    # compute the base
    sigmaS = max(np.shape(im)[0], np.shape(im)[1]) / 50
    sigmaR = 0.4
    base = bilateral_grid(lumLogged, sigmaS, sigmaR)
    # compute the detail
    detail = lumLogged - base
    # compute scale factor
    largeRange = np.max(base) - np.min(base)
    k = np.log10(targetBase) / largeRange
    # compute log output
    outLog = detailAmp * detail + k * (base - np.max(base))
    return 10**outLog * chrom












