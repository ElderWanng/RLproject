import math

import numpy as np
import matplotlib.pyplot as plt

import matplotlib as mp
import IPython
from solver import Solver
from base_obj import Base_Obj
class Quadrotor(Base_Obj):
    """
    This class describes a cart pole model and provides some helper functions
    """

    def __init__(self):
        """
        constructor of the class, takes as input desired discretization number
        for x (angle), v (angular velocity) and u (control) and the maximum control
        """
        # store discretization information
        super().__init__()
        self.mass = 0.500
        self.inertia = 0.1

        self.length = 0.15

        # gravity constant
        self.g = 9.81

        # integration step
        self.dt = 0.01

        self.ns = 6
        self.nu = 2




    def next_state(self, z, u,t):
        """
        Inputs:
        z: state of the cart pole syste as a numpy array (x,theta,v,omega)
        u: control as a scalar number

        Output:
        the new state of the pendulum as a numpy array
        """
        x = z[0]
        vx = z[1]
        y = z[2]
        vy = z[3]
        theta = z[4]
        omega = z[5]

        dydt = np.zeros([self.ns, ])
        dydt[0] = vx
        dydt[1] = (-(u[0] + u[1]) * np.sin(theta)) / self.mass
        dydt[2] = vy
        dydt[3] = ((u[0] + u[1]) * np.cos(theta) - self.mass * self.g) / self.mass
        dydt[4] = omega
        dydt[5] = (self.length * (u[0] - u[1])) / self.inertia

        z_next = z + dydt * self.dt

        return z_next

    def simulate(self, z0, controller, horizon_length, disturbance=False):
        """
        This function simulates the quadrotor for horizon_length steps from initial state z0

        Inputs:
        z0: the initial conditions of the quadrotor as a numpy array (x,vx,y,vy,theta,omega)
        controller: a function that takes a state z as argument and index i of the time step and returns a control u
        horizon_length: the horizon length

        disturbance: if True will generate a random push every seconds during the simulation

        Output:
        t[time_horizon+1] contains the simulation time
        z[4xtime_horizon+1] and u[1,time_horizon] containing the time evolution of states and control
        """
        t = np.zeros([horizon_length + 1, ])
        z = np.empty([self.ns, horizon_length + 1])
        z[:, 0] = z0
        u = np.zeros([self.nu, horizon_length])
        for i in range(horizon_length):
            u[:, i] = controller(z[:, i], i)
            z[:, i + 1] = self.next_state(z[:, i], u[:, i],i)
            if disturbance and np.mod(i, 100) == 0:
                dist = np.zeros([self.ns, ])
                dist[1::2] = np.random.uniform(-1., 1, (3,))
                z[:, i + 1] += dist
            t[i + 1] = t[i] + self.dt
        return t, z, u

    def animate_robot(self, x, u, dt=0.01):
        """
        This function makes an animation showing the behavior of the quadrotor
        takes as input the result of a simulation (with dt=0.01s)
        """

        min_dt = 0.1
        if (dt < min_dt):
            steps = int(min_dt / dt)
            use_dt = int(np.round(min_dt * 1000))
        else:
            steps = 1
            use_dt = int(np.round(dt * 1000))

        # what we need to plot
        plotx = x[:, ::steps]
        plotx = plotx[:, :-1]
        plotu = u[:, ::steps]

        fig = mp.figure.Figure(figsize=[8.5, 8.5])
        mp.backends.backend_agg.FigureCanvasAgg(fig)
        ax = fig.add_subplot(111, autoscale_on=False, xlim=[-4, 4], ylim=[-4, 4])
        ax.grid()

        list_of_lines = []

        # create the robot
        # the main frame
        line, = ax.plot([], [], 'k', lw=6)
        list_of_lines.append(line)
        # the left propeller
        line, = ax.plot([], [], 'b', lw=4)
        list_of_lines.append(line)
        # the right propeller
        line, = ax.plot([], [], 'b', lw=4)
        list_of_lines.append(line)
        # the left thrust
        line, = ax.plot([], [], 'r', lw=1)
        list_of_lines.append(line)
        # the right thrust
        line, = ax.plot([], [], 'r', lw=1)
        list_of_lines.append(line)

        def _animate(i):
            for l in list_of_lines:  # reset all lines
                l.set_data([], [])

            theta = plotx[4, i]
            x = plotx[0, i]
            y = plotx[2, i]
            trans = np.array([[x, x], [y, y]])
            rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

            main_frame = np.array([[-self.length, self.length], [0, 0]])
            main_frame = rot @ main_frame + trans

            left_propeller = np.array([[-1.3 * self.length, -0.7 * self.length], [0.1, 0.1]])
            left_propeller = rot @ left_propeller + trans

            right_propeller = np.array([[1.3 * self.length, 0.7 * self.length], [0.1, 0.1]])
            right_propeller = rot @ right_propeller + trans

            left_thrust = np.array([[self.length, self.length], [0.1, 0.1 + plotu[0, i] * 0.04]])
            left_thrust = rot @ left_thrust + trans

            right_thrust = np.array([[-self.length, -self.length], [0.1, 0.1 + plotu[0, i] * 0.04]])
            right_thrust = rot @ right_thrust + trans

            list_of_lines[0].set_data(main_frame[0, :], main_frame[1, :])
            list_of_lines[1].set_data(left_propeller[0, :], left_propeller[1, :])
            list_of_lines[2].set_data(right_propeller[0, :], right_propeller[1, :])
            list_of_lines[3].set_data(left_thrust[0, :], left_thrust[1, :])
            list_of_lines[4].set_data(right_thrust[0, :], right_thrust[1, :])

            return list_of_lines

        def _init():
            return _animate(0)

        ani = mp.animation.FuncAnimation(fig, _animate, np.arange(0, len(plotx[0, :])),
                                         interval=use_dt, blit=True, init_func=_init)
        plt.close(fig)
        plt.close(ani._fig)
        IPython.display.display_html(IPython.core.display.HTML(ani.to_html5_video()))

    def get_linearization(self, z, u,t):
        z = z.reshape(-1)
        dt = self.dt
        m = self.mass
        theta = z[4].item()
        omega = z[5].item()
        u0 = u[0].item()
        u1 = u[1].item()
        r = self.length
        I = self.inertia
        A = np.array([[1., dt, 0., 0., 0., 0.],
                      [0., 1., 0., 0., dt * (math.cos(theta) * (-u0 - u1)) / m, 0.],
                      [0., 0., 1., dt, 0., 0.],
                      [0., 0., 0., 1., -dt * (math.sin(theta) * (-u0 - u1)) / m, 0.],
                      [0., 0., 0., 0., 1., dt],
                      [0., 0., 0., 0., 0., 1.]])

        B = np.array([[0., 0.],
                      [-dt * (math.sin(theta) / m), -dt * (math.sin(theta) / m)],
                      [0., 0.],
                      [dt * (math.cos(theta) / m), dt * (math.cos(theta) / m)],
                      [0., 0.],
                      [(dt * r) / I, -(dt * r) / I]])

        return A, B
