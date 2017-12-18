#!/usr/bin/evn python

import os
import sys
import math
import pylab
import numpy as np

def plot_Focal_Loss(gama=2):
	'''FL(p_t) = -(1 - p_t)^gama * log(p_t)'''
	p_t = np.linspace(0,1,1000)
	y   = -np.power(1 - p_t, gama) * np.log(p_t)

	# compose plot
	pylab.title('Focal Loss')
	pylab.plot(p_t, y, 'co')      # same function with cyan dots
	pylab.plot(p_t, -np.log(p_t)) # softmax loss
	pylab.show() # show the plot

def plot_Gradient_of_Focal_Loss(gama=2):
	'''FL(p_t) = -(1 - p_t)^gama * log(p_t), here just for x instead of p_t'''
	p_t = np.linspace(0,1,1000)
	y   = np.power(1 - p_t, gama) * (gama * p_t * np.log(p_t) + p_t - 1) # if i == j

	# compose plot
	pylab.title('Gridient of Focal Loss')
	pylab.plot(p_t, y, 'co') # same function with cyan dots
	pylab.plot(p_t, p_t - 1) # softmax loss 
	pylab.show() # show the plot

if __name__ == '__main__':
	'''Loss and Gradient'''
	plot_Focal_Loss(gama=2)
	plot_Gradient_of_Focal_Loss(gama=2)
	
	pi = 0.01; bias = -np.log((1 - pi) / pi)
	print "pi:", pi, "bias:", bias
