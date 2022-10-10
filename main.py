import cv2
import numpy as np
import math
from matplotlib import pyplot as plt
import scipy

from external_energy import external_energy
from internal_energy_matrix import get_matrix

def click_event(event, x, y, flags, params):
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        #save point
        xs.append(x)
        ys.append(y)

        #display point
        cv2.circle(img, (x, y), 3, 128, -1)
        cv2.imshow('image', img)

def interpolate_points():
#    #spline = scipy.interpolate.CubicSpline(xs, ys)
#    #spline = scipy.interpolate.CubicHermitSpline(xs, ys)
#    spline = scipy.interpolate.splrep(xs, ys, s=0)
#    points_x = np.arange(min(xs), max(xs))
#    points_y = scipy.interpolate.splev(points_x, spline, der=0)
#
#    points = [(points_x[i], points_y[i]) for i in range(len(points_x))]
#    points.sort()
#    points = np.array(points)
#    points_x = points[:,0]
#    points_y = points[:,1]
#    print(points_x)
#    print(points_y)
#
#    plotted_image = plt.imread(img_path)
#    plot = plt.imshow(plotted_image, cmap='gray')
#
#    plt.scatter(points_x, points_y, c='r',s=1)
#    plt.pause(5)
    pass

def interpolate_circle_for_two_points():
    centre = ((xs[0]+xs[1])/2, (ys[0]+ys[1])/2)
    radius = np.linalg.norm(np.array([xs[0]-centre[0], ys[0]-centre[1]]))
    #print(radius)

    theta = np.linspace(0, 2*np.pi, 200)
    x = centre[0]+radius*np.cos(theta)
    y = centre[1]+radius*np.sin(theta)

    image = plt.imread(img_path)
    plot = plt.imshow(image, cmap='gray')
    plt.scatter(x, y, c='r', s=1)
    plt.pause(2)

    return x, y

def interpolate_external_energy():
    Ex = np.hstack((E[:,1:] - E[:,:-1], E[:,-1].reshape(1,E.shape[0]).T))

    Ey = np.vstack((E[:-1,:] - E[1:,:], E[-1,:]))

    #print("interpolate_external_energy: points_x ", points_x.flatten())
    #print("interpolate_external_energy: points_y ", points_y.flatten())

    fx = scipy.interpolate.interp2d(np.arange(0, img.shape[1]), np.arange(0, img.shape[0]), Ex)
    fx_values = fx(points_x.flatten(), points_y.flatten())
    fx_values = np.array([fx_values[i, i] for i in range(num_points)])
    #print("interpolate_external_energy: fx_values ", fx_values)

    fy = scipy.interpolate.interp2d(np.arange(0, img.shape[1]), np.arange(0, img.shape[0]), Ey)
    fy_values = fy(points_x.flatten(), points_y.flatten())
    fy_values = np.array([fy_values[i, i] for i in range(num_points)])
    #print("interpolate_external_energy: fy_values ", fy_values)

    return fx_values.reshape(num_points, 1), fy_values.reshape(num_points, 1)

if __name__ == '__main__':
    #point initialization
    img_path = '../images/circle.jpg'
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    temp = np.copy(img)
    #img = cv2.normalize(img.astype('float'), None, 0.0, 1.0, norm_type=cv2.NORM_MINMAX)

    xs = []
    ys = []
    cv2.imshow('image', img)

    cv2.setMouseCallback('image', click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    img = np.copy(temp)
    #selected points are in xs and ys

    #interpolate
    #implement part 1: interpolate between the  selected points
    points_x, points_y = interpolate_circle_for_two_points()

    alpha = 7
    beta = 0
    gamma = 2
    kappa = 1
    num_points = len(points_x)

    #get matrix
    M = get_matrix(alpha, beta, gamma, num_points)

    #get external energy
    w_line = 0.7 #############################################################
    w_edge = 0.7 #############################################################
    w_term = 0.1 #############################################################
    E = external_energy(img, w_line, w_edge, w_term)

    fx_values, fy_values = interpolate_external_energy()

    h,w = img.shape
    iteration_count = 0
    points_x = points_x.reshape(num_points, 1)
    points_y = points_y.reshape(num_points, 1)

    prev_points_x = np.copy(points_x)
    prev_points_y = np.copy(points_y)
    while True:
        #optimization loop
        iteration_count += 1
        fx_values, fy_values = interpolate_external_energy()

        if iteration_count > 200:
            break
        else:
            points_x = M @ (gamma*points_x - kappa*fx_values)
            print((fx_values[5], fy_values[5]))
            points_y = M @ (gamma*points_y - kappa*fy_values)
        points_x[points_x<0] = 0
        points_y[points_y<0] = 0
        points_x[points_x>w-1] = w-1
        points_y[points_y>h-1] = h-1
        print("X Difference: ", (points_x - prev_points_x).flatten())
        print("Y Difference: ", (points_y - prev_points_y).flatten())

        image = plt.imread(img_path)
        plot = plt.imshow(image, cmap='gray')
        plt.clf()
        plt.imshow(img, cmap='gray')
        #plt.imshow(img, cmap='binary')
        plt.scatter(points_x, points_y, c='g', s=1)
        #plt.plot(points_x, points_y, c='g')
        plt.pause(0.0005)
