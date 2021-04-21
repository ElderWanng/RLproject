from __future__ import print_function




import numpy as np
import matplotlib.pyplot as plt
import quadrotor
import math



robot = quadrotor.Quadrotor()
horizon_length = 1000
N = 1000
u_ = robot.mass*robot.g/2
z0 = np.array([1,0,0,0,0,0]).reshape([6,])
u0 = u_ * np.ones([2,1])
o = (2*math.pi/10)
r = 1
T = 10

middle = 500
tau = 0.2

# x_desired = [math.cos(o*(T/N)*i) for i in range(0,N+1)]
# y_desired = [r*math.sin(o*(T/N)*i) for i in range(0,N+1)]
# vx_desired = [-r*o*math.sin(o*(T/N)*i) for i in range(0,N+1)]
# vy_desired = [ r*o*math.cos(o*(T/N)*i) for i in range(0,N+1)]
# theta_desried = [0 for i in range(0,N+1)]
# omega_desried = [0 for i in range(0,N+1)]


# state_desired = np.array([x_desired,vx_desired,y_desired,vy_desired,theta_desried,omega_desried])
# state_desired = np.zeros_like(state_desired)
# #[6,N+1]
# state_desired[:,middle] = np.array([3,0,3,0,math.pi/2,0])
def generate_place(i):
    t = T/N*i
    return 3*math.e**(-abs(t-5)/tau)
def generateV(i):
    t = T/N*i
    if i<middle:
        return 3/tau*math.e**(-abs(t-5)/tau)
    elif i == middle:
        return 0
    else:
        return -3/tau*(math.e**(-abs(t-5)/tau))
def generate_theta(i):
    t = T/N*i
    return math.pi/2*math.e**(-abs(t-5)/tau)
def generate_omega(i):
    t = T/N*i
    if i<middle:
        return math.pi/2/tau*math.e**(-abs(t-5)/tau)
    elif i == middle:
        return 0
    else:
        return -math.pi/2/tau*math.e**(-abs(t-5)/tau)
x = [generate_place(i) for i in range(N+1)]
vx = [generateV(i) for i in range(N+1)]
y = [generate_place(i) for i in range(N+1)]   
vy = [generateV(i) for i in range(N+1)]
theta = [generate_theta(i) for i in range(N+1)]
omg = [generate_omega(i) for i in range(N+1)]
state_desired = np.array([x,vx,y,vy,theta,omg])

ref_traj = state_desired.T
z0 = np.zeros_like(state_desired[:,0])
u_init = [(robot.mass * robot.g / 2) * np.ones([2]) for _ in range(horizon_length)]
print(ref_traj.shape,state_desired.shape)


weight_mats = [100*np.diag([60,20,10,10,5,5])*(math.e**(-abs(i-middle)/50)) for  i in range(len(ref_traj))]
print(generateV(505))
q1c = 1
q2c = 1
tau1 = 20
tau2 = 60

tau3 = 10
tau4 = 10
q1 = np.array([1 if (350 < i) and (i < 5000) else 0 for i in range(N + 1)])
q2 = [q1c * (math.e ** (-abs(i - middle) / tau1)) for i in range(len(ref_traj))]

q4 = np.array([1 if (550 < i) and (i < 750) else 0 for i in range(N + 1)])
q5 = [q2c * (math.e ** (-abs(i - 600) / tau2)) for i in range(len(ref_traj))]
q6 = q4 * q5

q7 = np.array([1 if (700 < i) and (i < 5000) else 0 for i in range(N + 1)])
q8 = [q2c * (math.e ** (-abs(i - N) / tau3)) for i in range(len(ref_traj))]
q9 = q7 * q8

q10 = np.array([1 if (800 < i) and (i < 900) else 0 for i in range(N + 1)])
q11 = [q2c * (math.e ** (-abs(i - 856) / tau4)) for i in range(len(ref_traj))]
q12 = q10 * q11

q13 = np.array([1 if (600) and (i < 900) else 0 for i in range(N + 1)])

q = q1 * q2 + q6 + q9 + q12 + q13
q = np.clip(q)

856

weight_mats = [100 * np.diag([10, 0, 10, 0, 10000 * q[i], 0.5 * q[i]]) for i in range(len(ref_traj))]
# weight_mats2 = [100*np.diag([10,0,10,0,q2,0])*(math.e**(-abs(i-N)/tau2)) for  i in range(len(ref_traj))]
# weight_mats = [weight_mats[i]+weight_mats2[i] for i in range(len(weight_mats2))]
plt.figure()
plt.plot(np.sqrt(q))