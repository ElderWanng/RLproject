import math

import numpy as np
import matplotlib.pyplot as plt

import matplotlib as mp
import IPython
from solver import Solver

class Quadrotor:
    """
    This class describes a cart pole model and provides some helper functions
    """

    def __init__(self):
        """
        constructor of the class, takes as input desired discretization number
        for x (angle), v (angular velocity) and u (control) and the maximum control
        """
        # store discretization information
        self.mass = 0.500
        self.inertia = 0.1

        self.length = 0.15

        # gravity constant
        self.g = 9.81

        # integration step
        self.dt = 0.01

        self.ns = 6
        self.nu = 2
        self.ilqr_ = None

        return

    def build_lqr_solver(self,ref_pnts,weight_mats,A ,B, R ):
        self.T_ = len(ref_pnts)
        self.n_dims_ = len(ref_pnts[0])
        self.ref_array = np.copy(ref_pnts)
        self.weight_array = [mat for mat in weight_mats]
        self.A_ = A

        self.B_ = B
        self.R_ = R
        self.plant_dyn_ = self.next_state

        def tmp_cost_func(x, u, t, aux):
            err = x[0:self.n_dims_] - self.ref_array[t]
            # autograd does not allow A.dot(B)
            cost = np.dot(np.dot(err, self.weight_array[t]), err) + np.sum(u ** 2) * self.R_
            # if t > self.T_ - 1:
            #     # regularize velocity for the termination point
            #     # autograd does not allow self increment
            #     cost = cost + np.sum(x[self.n_dims_:] ** 2) * self.R_ * self.Q_vel_ratio_
            return cost

        self.cost_ = tmp_cost_func
        self.ilqr_ = Solver(horizion=self.T_ - 1, obj=self.plant_dyn_, cost=self.cost_,
                                            use_autograd=self.use_autograd)



        if not self.use_autograd:
            # self.plant_dyn_dx_ = lambda x, u, t, aux: self.A_
            # self.plant_dyn_du_ = lambda x, u, t, aux: self.B_
            #
            # def tmp_cost_func_dx(x, u, t, aux):
            #     err = x[0:self.n_dims_] - self.ref_array[t]
            #     grad = np.concatenate([2 * err.dot(self.weight_array[t]), np.zeros(self.n_dims_)])
            #     return grad
            #
            # self.cost_dx_ = tmp_cost_func_dx
            #
            # self.cost_du_ = lambda x, u, t, aux: 2 * self.R_ * u
            #
            # def tmp_cost_func_dxx(x, u, t, aux):
            #     hessian = np.zeros((2 * self.n_dims_, 2 * self.n_dims_))
            #     hessian[0:self.n_dims_, 0:self.n_dims_] = 2 * self.weight_array[t]
            #
            #     if t > self.T_ - 1:
            #         hessian[self.n_dims_:, self.n_dims_:] = 2 * np.eye(self.n_dims_) * self.R_ * self.Q_vel_ratio_
            #     return hessian
            #
            # self.cost_dxx_ = tmp_cost_func_dxx
            #
            # self.cost_duu_ = lambda x, u, t, aux: 2 * self.R_ * np.eye(self.n_dims_)
            # self.cost_dux_ = lambda x, u, t, aux: np.zeros((self.n_dims_, 2 * self.n_dims_))

            #build an iLQR solver based on given functions...
            self.ilqr_.plant_dyn_dx = self.A_
            self.ilqr_.plant_dyn_du = self.B_
            self.ilqr_.cost_dx = self.cost_dx_
            self.ilqr_.cost_du = self.cost_du_
            self.ilqr_.cost_dxx = self.cost_dxx_
            self.ilqr_.cost_duu = self.cost_duu_
            self.ilqr_.cost_dux = self.cost_dux_

        return










    def next_state(self, z, u):
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
            z[:, i + 1] = self.next_state(z[:, i], u[:, i])
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

    def get_linearization(self, z, u):
        assert z.shape == (6, 1)
        assert u.shape == (2, 1)
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
