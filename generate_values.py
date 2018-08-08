import numpy as np
import matplotlib.pyplot as plt
import math

def generate_true_ellipse(num_trials, dt, semimajor, semiminor):
    x = np.zeros((2, num_trials)) # x and y coordinates
    t = np.linspace(0, 2*math.pi, num_trials)
    for i in range(0, num_trials):
        x[0,i] = semimajor*np.cos(t[i])
        x[1,i] = semiminor*np.sin(t[i])

    return x

def generate_GPS_ellipse(num_trials, dt, semimajor, semiminor, std_dev_x, std_dev_y):
    x = generate_true_ellipse(num_trials,dt, semimajor, semiminor)
    t = np.linspace(0, 2*math.pi, num_trials)
    for i in range(0, num_trials):
        x[0,i] = x[0,i]+np.random.normal(0, std_dev_x)
        x[1,i] = x[1,i]+np.random.normal(0, std_dev_y)

    return x

def draw_ellipse(x):
    plt.scatter(x[0], x[1])
    plt.grid(color='lightgray',linestyle='--')
    plt.show()

draw_ellipse(generate_GPS_ellipse(200,0.01,200,100,3,3))