from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from numpy import linalg as LA

def get_instance(module, name, config, *args):
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])


def draw_error(targets, errors):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_aspect('equal')

    ratio = (errors - np.min(errors)) / (np.max(errors) - np.min(errors))
    colors = cm.rainbow(ratio)
    
    ax.scatter(targets[:,0], targets[:,1], targets[:,2], c=colors, marker='^')

    ax.set_xlim(-155, 155)
    ax.set_ylim(-155, 155)
    ax.set_zlim(0, 310)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    m = cm.ScalarMappable(cmap=cm.rainbow)
    m.set_array(errors)
    plt.colorbar(m)

    fig.savefig('visualize_3d_error.png')
    plt.show()
    plt.close(fig)


def visualize_2d(predictions, targets, num_spot):
    num_dots = predictions.shape[0]
    depths = np.array([33.0, 54.5, 119.0, 280.0])
    target_x = [[], [], [], []]
    target_y = [[], [], [], []]
    target_z = [[], [], [], []]
    prediction_x = [[], [], [], []]
    prediction_y = [[], [], [], []]
    prediction_z = [[], [], [], []]
    for i in range(num_dots):
        spot = np.argmin(abs(depths - targets[i][2])) # 0-based
        target_x[spot].append(targets[i][0])
        target_y[spot].append(targets[i][1])
        target_z[spot].append(targets[i][2])
        prediction_x[spot].append(predictions[i][0])
        prediction_y[spot].append(predictions[i][1])
        prediction_z[spot].append(predictions[i][2])

    # max_error = 0
    # for i in range(num_spot):
    #     errors = np.sqrt((np.array(target_x[i]) - np.array(prediction_x[i]))**2 + (np.array(target_y[i]) - np.array(prediction_y[i]))**2)
    #     max_error = max(max_error, max(errors))
    # last_errors = errors

    max_error = 0
    for i in range(num_spot):
        errors = np.zeros(len(target_x[i]))
        for j in range(len(target_x[i])):
            errors[j] = np.mean(compute_angle_error(np.array((target_x[i][j], target_y[i][j], target_z[i][j])), np.array((prediction_x[i][j], prediction_y[i][j], prediction_z[i][j]))))
        max_error = max(max_error, max(errors))
    errors_range = errors
    
    for i in range(num_spot):
        fig = plt.figure()
        # ########### Method 1: draw dots and arrows ###########
        # # plot four x-y planes
        # # draw dots
        # plt.plot(target_x[i], target_y[i], 'bo', markersize=1, markerfacecolor='none')
        # plt.plot(prediction_x[i], prediction_y[i], 'ro', markersize=1, markerfacecolor='none')
        # plt.axis([min(prediction_x[i]), max(prediction_y[i]), min(prediction_y[i]), max(prediction_y[i])])
        # # draw arrows
        # for index in range(len(target_x[i])):
        #     plt.annotate('', xytext=(target_x[i][index], target_y[i][index]), xy=(prediction_x[i][index], prediction_y[i][index]),
        #                  arrowprops=dict(fc='plum', ec='plum', shrink=0.05, width=0.005, headwidth=0.005, headlength=0.005),
        #     )

        ########### Method 2: draw dots with different colors ###########

        # # distance error
        # errors = np.sqrt((np.array(target_x[i]) - np.array(prediction_x[i]))**2 + (np.array(target_y[i]) - np.array(prediction_y[i]))**2)
        # angle error
        errors = np.zeros(len(target_x[i]))
        for j in range(len(target_x[i])):
            errors[j] = np.mean(compute_angle_error(np.array((target_x[i][j], target_y[i][j], target_z[i][j])), np.array((prediction_x[i][j], prediction_y[i][j], prediction_z[i][j]))))
        colors = cm.rainbow(errors/max_error)
        for j in range(len(target_x[i])):
            plt.plot(target_x[i][j], target_y[i][j], c=colors[j], marker='o', markersize=2)
        m = cm.ScalarMappable(cmap=cm.rainbow)
        m.set_array(errors_range)
        plt.colorbar(m)

        plt.title('Angle error in degree')
        plt.xlabel('x (degree)')
        plt.ylabel('y (degree)')
        fig.savefig('color_visualize/visualize_2d_xy_spot%d.png' % (i+1))
        plt.close(fig)

    # plot z-y plane
    fig = plt.figure()

    # ########### Method 1: draw dots and arrows ###########
    # plt.plot(targets[:,2], targets[:,1], 'bo', markersize=0.5)
    # plt.plot(predictions[:,2], predictions[:,1], 'ro', markersize=0.5)
    # plt.axis([0, 300, -60, 90]) # cm
    # # draw arrows
    # for index in range(num_dots):
    #     plt.annotate('', xytext=(targets[index, 2], targets[index, 1]), xy=(predictions[index,2], predictions[index,1]),
    #                  arrowprops=dict(fc='plum', ec='plum', shrink=0.05, width=0.005, headwidth=0.005, headlength=0.005),
    #     )

    ########### Method 2: draw dots with different colors ###########
    errors = abs(np.array(predictions[:,2]) - np.array(targets[:,2]))
    max_error = np.quantile(errors, 0.995)
    print(sum(errors>max_error))
    errors = np.minimum(errors, max_error)
    print('max_error', max_error)
    colors = cm.rainbow(errors/max_error)
    for j in range(num_dots):
        plt.plot(targets[j][2], targets[j][1], c=colors[j], marker='o', markersize=0.5)
    m = cm.ScalarMappable(cmap=cm.rainbow)
    m.set_array(errors)
    plt.colorbar(m)
    
    plt.xlabel('z (cm)')
    plt.ylabel('y (cm)')
    plt.title('Depth error in cm')
    fig.savefig('color_visualize/visualize_2d_yz.png')
#    plt.show()
    plt.close(fig)


    
def visualize_3d(predictions, targets):
    """Visualize 3D plots. """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_aspect('equal')

    ax.scatter(predictions[:,0], predictions[:,1], predictions[:,2], c='r', marker='o')
    ax.scatter(targets[:,0], targets[:,1], targets[:,2], c='b', marker='^')
    
    #plt.xlim(np.min(predictions[:,0])-10, np.max(predictions[:,0])+10)
    #plt.ylim(np.min(predictions[:,1])-10, np.max(predictions[:,1])+10)
    ax.set_xlim(-155, 155)
    ax.set_ylim(-155, 155)
    ax.set_zlim(0, 310)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    ax.view_init(azim=-90, elev=90)
    fig.savefig('visualize_3d_front.png')
    ax.view_init(azim=-90, elev=0)
    fig.savefig('visualize_3d_top.png')

    plt.show()
    plt.close(fig)
    

def compute_angle_error(p1, p2, origin=np.zeros(3)):
    """Compute angle error of two vectors.
    
    Keyword arguments:
    p1 -- point 1
    p2 -- point 2
    origin -- origin of coordinate
    """
    f = lambda vec1, vec2: np.arccos(np.dot(vec1, vec2) / LA.norm(vec1) / LA.norm(vec2)) / np.pi * 180 
    vec1 = p1 - origin
    vec2 = p2 - origin
    hori_error = abs(f(vec1, np.array([0, vec1[1], vec1[2]])) -
                     f(vec2, np.array([0,vec2[1],vec2[2]])))
    verti_error = abs(f(vec1, np.array([vec1[0], 0, vec1[2]])) -
                      f(vec2, np.array([vec2[0],0,vec2[2]])))
    return np.array([hori_error, verti_error])



    
