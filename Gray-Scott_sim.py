import numpy
import random
import matplotlib.pyplot as plt
import matplotlib.animation
matplotlib.rcParams['animation.embed_limit']=2**128
import scipy.signal

#parameters


f=0.0367
k=0.0649
'''
#worms
k=0.056
f=0.025


#mitosis
f=0.0367
k=0.0649

#coral growth
f=0.061
k=0.05207

'''

da=0.2
db=0.1

#kernel
kernel=numpy.array([[0.05,0.2,0.05],[0.2,-1.0,0.2],[0.05,0.2,0.05]],dtype=numpy.float64)

#grid size NxN
N=128

A=numpy.ones((N,N), dtype=numpy.float64)
B=numpy.zeros((N,N),dtype=numpy.float64)

def initialize_B(x):
    if (random.random()<0.05):
        return 1.0
    else:
        return 0.0

initialize_B_vf=numpy.vectorize(initialize_B)

B=initialize_B_vf(B)

fig= plt.figure()
im=plt.imshow(B, interpolation='none', cmap='PRGn',vmin=0,vmax=1)
fig.colorbar(im)

#amation function

def update(frame):
    global A,B

    AB2=A*B*B

    DA=scipy.signal.convolve2d(A,kernel,mode='same',boundary='wrap')
    DB = scipy.signal.convolve2d(B, kernel, mode='same', boundary='wrap')

    A+=(da*DA)-AB2+(1-A)*f
    B+=(db*DB)+AB2-(f+k)*B

    im.set_data(B)
    return

anim=matplotlib.animation.FuncAnimation(fig,update,frames=100, interval=50)
plt.show()