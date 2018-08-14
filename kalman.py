import numpy as np
import generate_values
import matplotlib.pyplot as plt


def kalman_filter(num_trials, semimajor, semiminor, std_dev_x, std_dev_y):
    """ kalman_filter() applies a basic Kalman filter on the simulated noisy output from generate_data.py. 
    The velocity is held constant at 2 m/s. This value can be changed in the generate_values.py source. 
    There is assumed to be no noise in the prediction step of the filter and hence, the process noise covariance matrix
    and the predicted state noise matrix are set to zero matrices.
    Measurements are taken at every 0.001 second (dt). This can be changed in the kalman.py source.
    """

    dt = 0.001
    dt_sq = dt**2
    # std_dev_x and std_dev_y of sensors is 3m
    # generate_noisy_values(num_trials, dt, std_dev_x, std_dev_y, x_init, y_init)
    noisy_readings = generate_values.generate_GPS_ellipse(num_trials,dt,semimajor,semiminor,std_dev_x,std_dev_y)   

    A = np.array([[1, 0, dt, 0],
                [0, 1, 0, dt],
                [0, 0, 1, 0 ],
                [0, 0, 0, 1 ]])

    B = np.array([[0.5*(dt_sq), 0],
                  [0, 0.5*(dt_sq)],
                  [dt, 0],
                  [0,dt]])
    H = np.eye(4)

    # Init var of x and y are large because uncertain of original position.
    # Vel_x and vel_y are 0.1.
    # All covariances are equal to 0 as each state var is assumed independent.

    P = np.array([ [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])
    
    variances = np.zeros((4,num_trials))
    # Q = np.zeros(4) # assuming no process noise
    Q = 0.01*np.dot(A, A.T)
    R = np.array([ [9, 0, 0, 0],
                    [0, 9, 0, 0],
                    [0, 0, 0.01, 0],
                    [0, 0, 0, 0.01]])
    # [x y v_x v_y].T
    state = np.zeros((4, num_trials))
    
    # initialize state (initial x, y, v_x, and v_y are 0)
    state[:, [0]] = np.array([[0, 0, 0, 0]]).T

    # u = np.array([[a_x],
    #               [a_y]])
    u = np.array([[1],[1]])

    for i in range(1, num_trials):
        state[:,[i]] = np.dot(A,state[:, [i - 1]]) +  np.dot(B,u) + np.zeros((4,1)) # the predicted state noise is set to 0
        # state[:,[i]] = np.dot(A,state[:, [i - 1]])
        P = np.dot(np.dot(A,P),A.T) + Q

        # gain
        K_num = np.dot(P, H.T)
        K_denom = np.dot(np.dot(H,P),H.T) + R
        K = np.dot(K_num,np.linalg.inv(K_denom))
    
        # Update
        innovation = noisy_readings[:,[i]] - np.dot(H,state[:,[i]])
        state[:,[i]] = state[:,[i]]+ np.vstack(np.dot(K,innovation))
        P = np.dot(np.eye(4) - np.dot(K,H),P)

        # more optimal filter error covar. update for larger number of trials
        ImKH = np.eye(4) - np.dot(K,H)
        P = np.dot(np.dot(ImKH, P), ImKH.T) + np.dot(np.dot(K,R),K.T)

        # save variances at each step
        varianceToSave = np.array([[P[0,0]],
                           [P[1,1]],
                           [P[2,2]],
                           [P[3,3]]])
        variances[:,[i]] = varianceToSave

    return state, variances


def plot_states(num_trials, semimajor, semiminor, std_dev_x, std_dev_y):
    """ Calls kalman_filter() and plots the prediction and correction at every time interval
    for both position and velocity. """

    states, _ = kalman_filter(num_trials, semimajor, semiminor, std_dev_x, std_dev_y)
    
    # fig = plt.figure(num=1,figsize=(12, 10), dpi=100)
    estimates = generate_values.generate_true_ellipse(num_trials,0.001,semimajor,semiminor)
    # plt.figure()
    plt.plot(states[0], states[1])
    plt.plot(estimates[0], estimates[1])

    # ax1 = fig.add_subplot(221)  # x values
    # plt.plot(states[0],label = 'Filtered values')
    # plt.plot(estimates[0],label = 'Predicted values')
    # handles, labels = ax1.get_legend_handles_labels()
    # ax1.set_title('x position')

    # ax2 = fig.add_subplot(222)  # y values
    # plt.plot(states[1])
    # plt.plot(estimates[1])
    # ax2.set_title('y position')

    # ax3 = fig.add_subplot(223) # v_x values
    # plt.plot(states[2])
    # plt.plot(estimates[2])
    # ax3.set_title('$v_x$ values')

    # ax4 = fig.add_subplot(224) # v_y values
    # plt.plot(states[3])
    # plt.plot(estimates[3])
    # ax4.set_title('$v_y$ values')


    # fig.legend(handles, labels)     # handles and labels for ax2, ax3, ax4 are same as for ax1
    # fig.subplots_adjust(hspace=.2,wspace=.2)
    
    plt.show()


def plot_variances(num_trials, semimajor, semiminor, std_dev_x, std_dev_y):
    """ Plots the variances of each state variable over time. The covariances are initally 0 
    in this model, and therefore aren't plotted."""
    _, variances = kalman_filter(num_trials, semimajor, semiminor, std_dev_x, std_dev_y)
    
    fig2 = plt.figure(num=2,figsize=(12,10),dpi=100)

    bx1 = fig2.add_subplot(221)
    plt.plot(variances[0])
    plt.yscale('log')
    bx1.set_title('${\sigma_x}^2$ over time')
    
    bx2 = fig2.add_subplot(222, sharex=bx1)
    plt.plot(variances[1])
    plt.yscale('log')
    bx2.set_title('${\sigma_y}^2$ over time')

    bx3 = fig2.add_subplot(223, sharex=bx1)
    plt.plot(variances[2])
    plt.yscale('log')
    bx3.set_title('${\sigma_{v_x}}^2$ over time')

    bx4 = fig2.add_subplot(224, sharex=bx1)
    plt.plot(variances[3])
    plt.yscale('log')
    bx4.set_title('${\sigma_{v_y}}^2$ over time')

    plt.show()


# num_trials=1000, x_init=2, y_init=2, a_x = a_y = 0.1
plot_states(1000,300,200, 300, 0)
# plot_variances(1000,2,2, 2, 2)
