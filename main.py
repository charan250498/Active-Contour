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
    xs.append(xs[0])
    ys.append(ys[0])
    n = len(xs)

    x = []
    y = []
    count_between_2_selected_points = 7

    for i in range(n-1): # as it should not include the last point which is the first point itself.
        points_in_x = np.linspace(xs[i], xs[i+1], count_between_2_selected_points)
        points_in_y = np.linspace(ys[i], ys[i+1], count_between_2_selected_points)
        #slope calculation
        if xs[i] == xs[i+1]:
            slope = "infinity"
        else:
            slope = (ys[i+1] - ys[i])/(xs[i+1]-xs[i])

        x.append(xs[i])
        y.append(ys[i])

        if slope == "infinity":
            for y_point in points_in_y:
                x_point = xs[i]
                x.append(xs[i])
                y.append(y_point)
        else:
            for x_point in points_in_x:
                if slope == 0:
                    y_point = ys[i]
                else:
                    y_point = slope*(x_point-xs[i]) + ys[i]
                x.append(x_point)
                y.append(y_point)

    return np.array(x), np.array(y)

def interpolate_circle_for_two_points():
    centre = ((xs[0]+xs[1])/2, (ys[0]+ys[1])/2)
    radius = np.linalg.norm(np.array([xs[0]-centre[0], ys[0]-centre[1]]))

    theta = np.linspace(0, 2*np.pi, 300)
    x = centre[0]+radius*np.cos(theta)
    y = centre[1]+radius*np.sin(theta)

    x[x<0] = 0
    y[y<0] = 0
    x[x>w-1] = w-1
    y[y>h-1] = h-1

    image = plt.imread(img_path)
    plot = plt.imshow(image, cmap='gray')
    plt.scatter(x, y, c='r', s=1)
    plt.pause(2)

    return x, y

def bilinear_interpolate(x, y, func):
    x1 = math.floor(x)
    x2 = math.ceil(x)
    y1 = math.floor(y)
    y2 = math.ceil(y)
    if (x2 == x1) and (y2 == y1):
        fxy = func[int(x),int(y)]
    elif x2 == x1:
        fxy = ((y1-y)*func[int(x),y2] + (y-y2)*func[int(x),y1])/(y1-y2)
    elif y2 == y1:
        fxy = ((x2-x)*func[x1, int(y)] + (x-x1)*func[x2, int(y)])/(x2-x1)
    else:
        fxy2 = ((x2-x)*func[x1, y2] + (x-x1)*func[x2, y2])/(x2-x1)
        fxy1 = ((x2-x)*func[x1, y1] + (x-x1)*func[x2, y1])/(x2-x1)

        fxy = ((y1-y)*fxy2 + (y-y2)*fxy1)/(y1-y2)
    return fxy

def interpolate_external_energy():
    Ex = np.hstack((E[:,1:] - E[:,:-1], E[:,-1].reshape(1,E.shape[0]).T))
    Ey = np.vstack((E[:-1,:] - E[1:,:], E[-1,:]))

    fx_values = []
    fy_values = []

    for i in range(num_points):
        fx_values.append(bilinear_interpolate(points_y[i], points_x[i], Ex))
        fy_values.append(bilinear_interpolate(points_y[i], points_x[i], Ey))

    fx_values = np.array(fx_values)
    fy_values = np.array(fy_values)

    return fx_values.reshape(num_points, 1), fy_values.reshape(num_points, 1)

if __name__ == '__main__':
    #point initialization
    img_path = '../images/circle.png'
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    temp = np.copy(img)

    h,w = img.shape

    xs = []
    ys = []
    cv2.imshow('image', img)

    cv2.setMouseCallback('image', click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    img = np.copy(temp)
    img = cv2.GaussianBlur(img, (5,5), 0)
    #selected points are in xs and ys

    #interpolate
    #implement part 1: interpolate between the  selected points
    points_x, points_y = interpolate_points() ############################# FOR OTHERS ###############################
    #points_x, points_y = interpolate_circle_for_two_points() ############################ JUST FOR CIRCLE #############################

    alpha = 1.0
    beta = 0.2
    gamma = 0.4
    kappa = 0.6
    num_points = len(points_x)

    #get matrix
    M = get_matrix(alpha, beta, gamma, num_points)

    #get external energy
    w_line = 0.001
    w_edge = 0.2
    w_term = 0.1 
    E = external_energy(img, w_line, w_edge, w_term)

    max_iter_count = 400

    fx_values, fy_values = interpolate_external_energy()

    iteration_counter = 0
    points_x = points_x.reshape(num_points, 1)
    points_y = points_y.reshape(num_points, 1)

    while True:
        #optimization loop
        iteration_counter += 1
        points_x = points_x.reshape(num_points)
        points_y = points_y.reshape(num_points)
        fx_values, fy_values = interpolate_external_energy()
        points_x = points_x.reshape(num_points, 1)
        points_y = points_y.reshape(num_points, 1)

        if iteration_counter > max_iter_count:
            break
        else:
            points_x = M @ (gamma*points_x - kappa*fx_values)
            points_y = M @ (gamma*points_y - kappa*fy_values)

        points_x[points_x<0] = 0
        points_y[points_y<0] = 0
        points_x[points_x>w-1] = w-1
        points_y[points_y>h-1] = h-1

        image = plt.imread(img_path)
        plt.clf()
        plt.imshow(img, cmap='gray')
        plt.scatter(points_x, points_y, c='g', s=3)
        plt.pause(0.0001)
