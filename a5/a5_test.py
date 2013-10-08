import sys
sys.path.append('utils')
import imageIO as io
import numpy as np
import a5
import multiprocessing as multi



def test_computeWeight():
  im=io.imread('data/design-2.png')
  out=a5.computeWeight(im)
  io.imwrite(out, 'design-2_mask.png')

def test_computeFactor():
  im2=io.imread('data/design-2.png')
  im3=io.imread('data/design-3.png')
  w2=a5.computeWeight(im2)
  w3=a5.computeWeight(im3)
  out=a5.computeFactor(im2, w2, im3, w3)
  if abs(out-50.8426955376)<1 : 
    print 'Correct'
  print out
  
def test_makeHDR():
  import glob, time
##  p=multi.Pool(processes=8)
##  im_list=p.map(io.imread,inputs)
  inputs=glob.glob('data/ante3-*.png')
  im_list = []
  for inp in inputs:
    im_list.append(io.imread(inp))


  hdr=a5.makeHDR(im_list)
  np.save('hdr', hdr)

  hdr_scale=hdr/max(hdr.flatten())
  io.imwrite(hdr_scale, 'hdr_linear_scale.png')
  

def test_toneMap():
  hdr=np.load('hdr.npy')
  out=a5.toneMap(hdr, useBila=True)
  io.imwrite(out, 'tone_map.png')


# Uncomment the below to test your code

##test_computeWeight()
##test_computeFactor()
##test_makeHDR()
test_toneMap()

