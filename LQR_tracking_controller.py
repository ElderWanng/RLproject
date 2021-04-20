"""
LQR based trajectory controller
"""
from __future__ import print_function

import pylqr
import solver
try:
    import jax.numpy as np
except ImportError:
    import numpy as np

class LQR_Track_Controller:
    def __init__(self,R,dt,use_autograd = False):
        self.aux = None
        self.R_ = R
        self.dt_ = dt

        #desired functions for plant dynamics and cost
        self.plant_dyn_ = None
        self.plant_dyn_dx_ = None
        self.plant_dyn_du_ = None

        self.cost_ = None
        self.cost_dx_ = None
        self.cost_du_ = None
        self.cost_dxx_ = None
        self.cost_duu_ = None
        self.cost_dux_ = None

        self.lqr_ = None

        self.use_autograd=use_autograd
        return

    def build_LQR_tracking(self,ref_pnts, weight_mats,obj):
        #obj must have next_state,  get_linearization
        self.obj = obj
        #figure out dimension
        self.T_ = len(ref_pnts)# leave the last point alone


        self.ref_array = np.copy(ref_pnts)
        self.weight_array = [mat for mat in weight_mats]
        #clone weight mats if there are not enough weight mats
        for i in range(self.T_ - len(self.weight_array)):
            self.weight_array.append(self.weight_array[-1])


        def tmp_cost_func(x, u, t, aux):
            err = x[0:self.obj.nu] - self.ref_array[t]
            #autograd does not allow A.dot(B)
            cost = np.dot(np.dot(err, self.weight_array[t]), err) + np.sum(u**2) * self.R_
            return cost

        self.cost_ = tmp_cost_func
        self.lqr_ = solver.Solver(T=self.T_ - 1, plant_dyn=self.plant_dyn_, cost=self.cost_,
                                  use_autograd=self.use_autograd)
        if not self.use_autograd:
            self.plant_dyn = lambda x, u, t, aux:obj.next_state(x,u,t)
            self.plant_dyn_du_ = lambda x, u, t, aux: obj.get_linearization(x, u, t)[0]
            self.plant_dyn_dx_ = lambda x, u, t, aux: obj.get_linearization(x, u, t)[1]
            def tmp_cost_func_dx(x, u, t, aux):
                err = x - self.ref_array[t]
                grad = np.concatenate([2 * err.dot(self.weight_array[t])])
                return grad

            self.cost_dx_ = tmp_cost_func_dx

            self.cost_du_ = lambda x, u, t, aux: 2 * self.R_ * u

            def tmp_cost_func_dxx(x, u, t, aux):
                hessian = np.zeros((self.obj.ns,self.obj.ns))
                hessian = 2 * self.weight_array[t]
                return hessian

            self.cost_dxx_ = tmp_cost_func_dxx

            self.cost_duu_ = lambda x, u, t, aux: 2 * self.R_ * np.eye(self.obj.nu)
            self.cost_dux_ = lambda x, u, t, aux: np.zeros((self.obj.nu, 2 * self.obj.nu))

            # build an iLQR solver based on given functions...
            self.lqr_.plant_dyn = self.plant_dyn
            self.lqr_.plant_dyn_dx = self.plant_dyn_dx_
            self.lqr_.plant_dyn_du = self.plant_dyn_du_
            self.lqr_.cost_dx = self.cost_dx_
            self.lqr_.cost_du = self.cost_du_
            self.lqr_.cost_dxx = self.cost_dxx_
            self.lqr_.cost_duu = self.cost_duu_
            self.lqr_.cost_dux = self.cost_dux_
        return

    def synthesize_trajectory(self,x0, u_array=None, n_itrs=50, tol=1e-6, verbose=True):
        if self.lqr_ is None:
            print('No iLQR solver has been prepared.')
            return None
        #initialization doesn't matter as global optimality can be guaranteed?
        if u_array is None:
            u_init = [np.zeros(self.obj.nu) for i in range(self.T_-1)]
        else:
            u_init = u_array
        x_init = np.zeros(self.obj.ns)
        x_init[:len(x0)] = x0

        # res = self.lqr_.ilqr_iterate(x_init, u_init, n_itrs=n_itrs, tol=tol, verbose=verbose)
        res_dict = self.lqr_.LQR_solve(x_init,u_init)
        x_star = res_dict['x_array_star']
        Ks = res_dict['K_array_opt']
        ks = res_dict['k_array_opt']
        x_array_new, u_array_new = self.apply_control(x_star, u_init, ks, Ks,alpha=1)
        return x_array_new[:, 0:self.obj.nu]

    def controller(self,u,x,k_array,K_array):
        pass

    def apply_control(self, x_array, u_array, k_array, K_array,alpha=1) :
        x_new_array = [None] * len(x_array)
        u_new_array = [None] * len(u_array)
        x_new_array[0] = x_array[0]
        for t in range(self.T_ -1 ):
            u_new_array[t] = u_array[t] + alpha * (k_array[t] + K_array[t].dot(x_new_array[t]-x_array[t]))
            x_new_array[t+1] = self.obj.next_state(x_new_array[t], u_new_array[t], t)
        return np.array(x_new_array), np.array(u_new_array)

if __name__ == '__main__':
    from base_obj import Base_Obj
    class test_obj(Base_Obj):
        def __init__(self):
            super(test_obj, self).__init__()
            self.dt = 0.1
            self.ns = 4
            self.nu = 2
            self.A_ = np.eye(2* 2)
            self.A_[0:2, 2:] = np.eye(2) * self.dt

            self.B_ = np.zeros((2*2,2))
            self.B_[2:, :] = np.eye(2) * self.dt

        def next_state(self,x,u,t):
            return np.dot(self.A_, x) + np.dot(self.B_, u)
        def get_linearization(self,x,u,t):
            return self.A_,self.B_


    dyn_plant = test_obj()
    def tracking_test():
        import matplotlib.pyplot as plt
        n_pnts = 200
        x_coord = np.linspace(0.0, 2 * np.pi, n_pnts)
        y_coord = np.sin(x_coord)
        vx = np.zeros_like(x_coord)
        vy = np.zeros_like(x_coord)
        ref_traj = np.array([x_coord, y_coord,vx,vy]).T
        weight_mats = [np.diag([1,1,0,0]) * 100]
        fig = plt.figure()
        ax = fig.add_subplot(111)
        # ax.hold(True)
        ax.plot(ref_traj[:, 0], ref_traj[:, 1], '.-k', linewidth=3.5)
        ax.plot([ref_traj[0, 0]], [ref_traj[0, 1]], '*', markersize=16)

        lqr_traj_ctrl = LQR_Track_Controller(R=.01, dt=0.01)
        lqr_traj_ctrl.build_LQR_tracking(ref_traj, weight_mats, dyn_plant)
        n_queries = 5
        for _ in range(n_queries):
            # start from a perturbed point
            x0 = ref_traj[0, :] + np.random.rand(4) * 2 - 1
            syn_traj = lqr_traj_ctrl.synthesize_trajectory(x0)
            # plot it
            ax.plot(syn_traj[:, 0], syn_traj[:, 1], linewidth=3.5)
        plt.show()


    tracking_test()




