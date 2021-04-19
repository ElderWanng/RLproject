from __future__ import print_function
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

    def __init__(self,horizion, obj, cost,use_autograd=True, constraint=None):
        self.horiztion = horizion
        self.obj = obj
        self.cost = cost
        self.constraint = constraint
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

            if constraint is not None:
                self.constraints_dx = jacobian(self.constraints, 0)
                self.constraints_du = jacobian(self.constraints, 1)
                self.constraints_dxx = jacobian(self.constraints_dx, 0)
                self.constraints_duu = jacobian(self.constraints_du, 1)
                self.constraints_dux = jacobian(self.constraints_du, 0)
        return

