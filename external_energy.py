from weakref import CallableProxyType
import numpy as np
from matplotlib import pyplot as plt
import cv2

def line_energy(image):
    #implement line energy (i.e. image intensity)
    #print("Image shape: ",image.shape)
    #eline = cv2.GaussianBlur(image, (15,15), cv2.BORDER_DEFAULT) ##################################
    eline = image.copy()
    return eline

def edge_energy(image):
    #implement edge energy (i.e. gradient magnitude)
    height, width = image.shape
    gx = np.hstack((image[:,1:] - image[:,:-1], image[:,-1].reshape(1,height).T)) # Reshape is used to convert it to a 2D array from 1D array.
    gy = np.vstack((image[:-1,:] - image[1:,:], image[-1,:]))

    eedge = -1*((gx**2+gy**2)**0.5)
    #eedge = -1*(gx**2+gy**2)
    #eedge = cv2.GaussianBlur(eedge, (3,3), 0)

    return eedge

def term_energy(image):
    #implement term energy (i.e. curvature)
    height, width = image.shape
    cxx = image[:,2:]-2*image[:,1:-1]+image[:,:-2] # short by two columns
    cxx = np.hstack(((image[:,1]-image[:,0]).reshape((1,height)).T, cxx))
    cxx = np.hstack((cxx, (image[:,-1]-image[:,-2]).reshape((1,height)).T))

    cyy = image[:-2,:]-2*image[1:-1,:]+image[2:,:] # short by two rows
    cyy = np.vstack((image[0,:]-image[1,:], cyy))
    cyy = np.vstack((cyy, image[-2,:]-image[-1,:]))

    cx = np.hstack((image[:,1:] - image[:,:-1], image[:,-1].reshape(1,height).T))
    cy = np.vstack((image[:-1,:] - image[1:,:], image[-1,:]))

    cxy = np.vstack((cx[:-1,:] - cx[1:,:], cx[-1,:]))

    eterm = ((cxx*(cy**2))-(2*cxy*cx*cy)+(cyy*(cx**2)))/((cx**2+cy**2 + 1)**1.5) ####################################################
    #eterm[np.logical_not(eterm > 1)] = 100 ##########################################################################################

    return eterm

def external_energy(image, w_line, w_edge, w_term):
    #implement external energy
    eline = line_energy(image)
    #plt.imshow(eline, cmap='gray')
    #plt.pause(5)
    eedge = edge_energy(image)
    #plt.imshow(eedge, cmap='gray')
    #plt.pause(5)
    eterm = term_energy(image)
    #plt.imshow(eterm, cmap='gray')
    #plt.pause(5)
    e_energy = w_line*eline+w_edge*eedge+w_term*eterm
    print("******external Energy*****")
    print(np.min(e_energy))
    print(np.max(e_energy))
    print("**********************")
    return e_energy