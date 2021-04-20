class Base_Obj:
    def __init__(self):
        self.ns = 0
        self.nu = 0
        self.dt = 0

    def next_state(self,x,u,t):
        raise NotImplementedError
    def get_linearization(self,x,u,t):
        raise NotImplementedError