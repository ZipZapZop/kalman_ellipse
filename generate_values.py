import numpy as np
import matplotlib.pyplot as plt
import math

def generate_true_ellipse(num_trials, dt, semimajor, semiminor):
    ''' Generates an ellipse based on parametric equations of an ellipse '''
    x = np.zeros((4, num_trials)) # x, y, v_x, v_y 
    t = np.linspace(0, 2*math.pi, num_trials)
    x[:,0] = np.array([semimajor*np.cos(t[0]),semiminor*np.sin(t[0]),0,0])
    for i in range(1, num_trials):
        x[0,i] = semimajor*np.cos(t[i])
        x[1,i] = semiminor*np.sin(t[i])
        x[2,i] = (x[0,i]-x[0,i-1])/dt
        x[3,i] = (x[1,i]-x[1,i-1])/dt
        
    return x

def generate_GPS_ellipse(num_trials, dt, semimajor, semiminor, std_dev_x, std_dev_y):
    ''' Adds noise to generate_true_ellipse() values '''
    x = generate_true_ellipse(num_trials, dt, semimajor, semiminor)
    for i in range(0, num_trials):
        x[0,i] = x[0,i]+np.random.normal(0, std_dev_x)
        x[1,i] = x[1,i]+np.random.normal(0, std_dev_y)
        x[2,i] = abs((x[0,i]-x[0,i-1]))
        x[3,i] = abs((x[1,i]-x[1,i-1]))
        # x[4,i] = math.sqrt(x[2,i]**2 + x[3,i]**2)

    return x

def draw_ellipse(x):
    t = np.linspace(0, 2*math.pi, 1000)
    plt.figure(1)
    plt.scatter(x[0], x[1])
    plt.title('Elliptical track')

    plt.figure(2)
    plt.plot(t,x[2],label='calculated $v_x$')
    plt.plot(t,x[3],label='calculated $v_y$')
    plt.grid(color='lightgray',linestyle='--')
    plt.legend()

    plt.show()

# draw_ellipse(generate_GPS_ellipse(1000,0.001,200,100,3,3))
