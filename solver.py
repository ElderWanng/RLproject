from __future__ import print_function
from typing import Callable, Iterator, Union, Optional, List
try:
    #note autograd should be replacable by jax in future
    # import autograd.numpy as np
    import jax.numpy as np
    from jax import grad, jacobian
    has_autograd = True
except ImportError:
    import numpy as np
    has_autograd = False

class Solver:
    def __init__(self,T, plant_dyn, cost,use_autograd=True, constraints=None):
        self.aux = None
        self.T = T
        self.plant_dyn = plant_dyn
        self.cost = cost
        self.constraints = constraints
        self.use_autograd = has_autograd and use_autograd

        self.plant_dyn_dx = None        #Df/Dx
        self.plant_dyn_du = None        #Df/Du
        self.cost_dx = None             #Dl/Dx
        self.cost_du = None             #Dl/Du
        self.cost_dxx = None            #D2l/Dx2
        self.cost_duu = None            #D2l/Du2
        self.cost_dux = None            #D2l/DuDx

        # self.constraints_dx = None      #Dc/Dx
        # self.constraints_du = None      #Dc/Du
        # self.constraints_dxx = None     #D2c/Dx2
        # self.constraints_duu = None     #D2c/Du2
        # self.constraints_dux = None     #D2c/DuDx

        self.constraints_lambda = 1000
        self.finite_diff_eps = 1e-5
        self.reg = .1
        self.alpha_array = np.array([0.5**i for i in range(10)])
        if self.use_autograd:
            #generate gradients and hessians using autograd
            #note in this case, the plant_dyn, cost and constraints must be specified with the autograd numpy
            self.plant_dyn_dx = jacobian(self.plant_dyn, 0)  #with respect the first argument   x
            self.plant_dyn_du = jacobian(self.plant_dyn, 1)  #with respect to the second argument   u

            self.cost_dx = grad(self.cost, 0)
            self.cost_du = grad(self.cost, 1)
            self.cost_dxx = jacobian(self.cost_dx, 0)
            self.cost_duu = jacobian(self.cost_du, 1)
            self.cost_dux = jacobian(self.cost_du, 0)

            if constraints is not None:
                self.constraints_dx = jacobian(self.constraints, 0)
                self.constraints_du = jacobian(self.constraints, 1)
                self.constraints_dxx = jacobian(self.constraints_dx, 0)
                self.constraints_duu = jacobian(self.constraints_du, 1)
                self.constraints_dux = jacobian(self.constraints_du, 0)
        return

    def LQR_solve(self,x0,u_init):
        horizon = len(u_init)
        x_array = self.forward_propagation(x0, u_init)
        u_array = np.copy(u_init)
        k_array, K_array = self.back_propagation(x_array, u_array)
        res_dict = {
        'x_array_star':np.array(x_array),
        'u_array_star':np.array(u_array),
        'k_array_opt':np.array(k_array),
        'K_array_opt':np.array(K_array)
        }
        return res_dict



    def forward_propagation(self, x0, u_array):
        """
        Apply the forward dynamics to have a trajectory starting from x0 by applying u

        u_array is an array of control signal to apply
        """
        traj_array = [x0]

        for t, u in enumerate(u_array):
            traj_array.append(self.plant_dyn(traj_array[-1], u, t, self.aux))

        return traj_array

    def back_propagation(self, x_array, u_array):

        u_array_sup = np.vstack([u_array, np.zeros(len(u_array[0]))])
        lqr_sys = self.build_lqr_system(x_array, u_array_sup)

        #k and K
        fdfwd = [None] * self.T
        fdbck_gain = [None] * self.T
        fdfwd = [None] * self.T
        fdbck_gain = [None] * self.T
        Vxx = lqr_sys['dldxx'][-1] #QN
        Vx = lqr_sys['dldx'][-1]  #qn
        for t in reversed(range(self.T)):
            #note to double check if we need the transpose or not
            Qx = lqr_sys['dldx'][t] + lqr_sys['dfdx'][t].T.dot(Vx) #qn + At
            Qu = lqr_sys['dldu'][t] + lqr_sys['dfdu'][t].T.dot(Vx) # Bt+ Bt
            Qxx = lqr_sys['dldxx'][t] + lqr_sys['dfdx'][t].T.dot(Vxx).dot(lqr_sys['dfdx'][t]) #Q + At Plast A
            Qux = lqr_sys['dldux'][t] + lqr_sys['dfdu'][t].T.dot(Vxx).dot(lqr_sys['dfdx'][t]) #0 + Bt Plast A
            Quu = lqr_sys['dlduu'][t] + lqr_sys['dfdu'][t].T.dot(Vxx).dot(lqr_sys['dfdu'][t]) #R + Bt Plast B

            #use regularized inverse for numerical stability
            # inv_Quu = self.regularized_persudo_inverse_(Quu, reg=self.reg)
            inv_Quu = np.linalg.inv(Quu)

            #get k and K
            fdfwd[t] = -inv_Quu.dot(Qu) # -inv(R + Bt Plast B)(B+ Bt )
            fdbck_gain[t] = -inv_Quu.dot(Qux) #-(R + Bt Plast B)^-1  (Bt Plast A)

            #update value function for the previous time step
            Vxx = Qxx - fdbck_gain[t].T.dot(Quu).dot(fdbck_gain[t]) #Q + At Plast A -
            Vx = Qx - fdbck_gain[t].T.dot(Quu).dot(fdfwd[t]) #this is pn+1
        return fdfwd, fdbck_gain

    def build_lqr_system(self, x_array, u_array):
        dfdx_array = []
        dfdu_array = []
        dldx_array = []
        dldu_array = []
        dldxx_array = []
        dldux_array = []
        dlduu_array = []
        for t, (x, u) in enumerate(zip(x_array, u_array)):
            dfdx_array.append(self.plant_dyn_du(x, u, t, self.aux))
            dfdu_array.append(self.plant_dyn_dx(x, u, t, self.aux))
            dldx_array.append(self.cost_dx(x, u, t, self.aux))
            dldu_array.append(self.cost_du(x, u, t, self.aux))
            dldxx_array.append(self.cost_dxx(x, u, t, self.aux))
            dlduu_array.append(self.cost_duu(x, u, t, self.aux))
            dldux_array.append(self.cost_dux(x, u, t, self.aux))
        lqr_sys = {
            'dfdx':dfdx_array,
            'dfdu':dfdu_array,
            'dldx':dldx_array,
            'dldu':dldu_array,
            'dldxx':dldxx_array,
            'dlduu':dlduu_array,
            'dldux':dldux_array
            }
        return lqr_sys

    def apply_control(self,x_array,u_array,ks,Ks,alpha=1):
        x_new_array = [None] * len(x_array)
        u_new_array = [None] * len(u_array)
        x_new_array[0] = x_array[0]
        for t in range(self.T):
            u_new_array[t] = u_array[t] + alpha * (ks[t] + Ks[t].dot(x_new_array[t] - x_array[t]))
            x_new_array[t + 1] = self.plant_dyn(x_new_array[t], u_new_array[t], t, self.aux)

        return np.array(x_new_array), np.array(u_new_array)







