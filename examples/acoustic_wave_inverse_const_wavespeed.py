"""
  Physics-Informed Neural Network 
  for solving the wave equation,
  where c is treated as unknown.

  u_xx = 1/c^2 u_tt, domain = (0, 2 pi)
  u(x,t) = sin(x - ct), analytical solution
  u(0,t) = u (2 pi,t), periodic boundary condition
  u(x,0) = sin(x), initial condition
  u_t(x,0) = -c cos(x)
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import math
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf 
print(tf.__version__)
np.set_printoptions(precision=3, suppress=True)

class PDE_PINN():

    # initialize the class; similar to constructor in C++
    # ---------------------------------------------------
    def __init__(self, 
                 activation_func = 'tanh',
                 domain = (0, 1, 0, 1), 
                 wave_speed_exact = 1,
                 wave_speed_ig = 1.5, # initial guess
                 n_collocation_pts = 2000,                 
                 n_initcond_pts = 128,
                 n_boundary_pts = 128,
                 n_observation_pts = 500):
        
        self.activation = activation_func
        self.domain = domain
        self.n_collocation_pts = n_collocation_pts
        self.n_boundary_pts = n_boundary_pts
        self.n_initcond_pts = n_initcond_pts
        self.n_observation_pts = n_observation_pts
        self.c_exact = wave_speed_exact
        self.c = tf.Variable([wave_speed_ig], dtype=tf.float32)

    # define neural network
    # ---------------------
    def neural_net(self, nn_arch=[16,8], input_shape=2):

        # 0. sequential layers (do non-sequential layers exist?)
        """
        Sequential groups a linear stack of layers into a tf.keras.Model.
        A Sequential model is appropriate for a plain stack of layers 
        where each layer has exactly one input tensor and one output tensor.
        """
        model = tf.keras.Sequential()
        
        # 1. input layer
        """
        In Keras, the input layer itself is not a layer, but a tensor. 
        It's the starting tensor you send to the first hidden layer. 
        This tensor must have the same shape as your training data.
        """
        model.add(tf.keras.layers.InputLayer(input_shape=input_shape, name='input'))
        
        # 2. hidden layers
        for n_nodes in nn_arch:
            print('- creating hidden layer with size: ', n_nodes)
            model.add(tf.keras.layers.Dense(n_nodes, activation=self.activation))
        
        # 3. output layer 
        model.add(tf.keras.layers.Dense(1, activation=None, name='output'))
        
        # 4.
        model.summary()
        self.nn_model = model

    # compute u(x,t) via neural net
    # -----------------------------
    @tf.function # Compiles a function into a callable TensorFlow graph.
    def compute_u(self, x, t): 
        u = self.nn_model(tf.concat([x, t], 1))
        return u

    # compute du/dt via neural net
    # ----------------------------
    @tf.function
    def compute_u_t(self, x, t): 
        u    = self.compute_u(x, t)
        u_t  = tf.gradients(u, t)[0]
        return u_t
        
    # residual of the PDE    
    # -------------------
    @tf.function
    def pde_residual(self, x, t):
        """
        PDE residual:= u_xx - 1/c^2 * u_tt
        """
        u    = self.compute_u(x, t)
        u_x  = tf.gradients(u, x)[0]
        u_xx = tf.gradients(u_x, x)[0]
        u_t  = tf.gradients(u, t)[0]
        u_tt = tf.gradients(u_t, t)[0]
        f    = u_xx - 1.0/(self.c * self.c) * u_tt
        return f

    def boundary_points(self):
        """
        Returns a batch of boundary points for training the PINN.
        Points are returned as a tuple (x, t, u).
        """
        # generate random numbers, if rand < 0.5, then x = x_l; otherwise x = x_r.
        x = np.where(np.random.rand(self.n_boundary_pts, 1) < 0.5, self.domain[0], self.domain[1])
        
        t = np.linspace(self.domain[2], self.domain[3], num=self.n_boundary_pts).reshape(self.n_boundary_pts, 1)            
        #u = np.zeros_like(x)
        u = np.sin(x - self.c * t)

        return (tf.convert_to_tensor(x, dtype=tf.float32),
                tf.convert_to_tensor(t, dtype=tf.float32),
                tf.convert_to_tensor(u, dtype=tf.float32))
    
    @tf.function
    def initial_condition_points (self, ic_func=None):
        """
        Returns a batch of initial condition points for training the PINN.
        Points are returned as a tuple (x, t, u).
        """

        x = np.linspace(self.domain[0], self.domain[1], num=self.n_initcond_pts).reshape(self.n_initcond_pts, 1)
        t = np.full_like(x, self.domain[2]) # corresponds to t = t_0
        u = np.sin(x)
        u_t = -self.c * np.cos(x)

        return (tf.convert_to_tensor(x, dtype=tf.float32),
                tf.convert_to_tensor(t, dtype=tf.float32),
                tf.convert_to_tensor(u, dtype=tf.float32),
                tf.convert_to_tensor(u_t, dtype=tf.float32))

    def collocation_points(self):
        """
        Returns a batch of random collocation points for training the PINN.
        """
        
        # randomly sampled locations in x-t
        x = np.random.uniform(low=self.domain[0], high=self.domain[1], size=(self.n_collocation_pts, 1))
        t = np.random.uniform(low=self.domain[2], high=self.domain[3], size=(self.n_collocation_pts, 1))
        return (tf.convert_to_tensor(x, dtype=tf.float32),
                tf.convert_to_tensor(t, dtype=tf.float32))

    def observation_points(self):
        # observation points between measured data and computed solution
        # randomly sampled points in x-t
        x = np.random.uniform(low=self.domain[0], high=self.domain[1], size=(self.n_observation_pts, 1))
        t = np.random.uniform(low=self.domain[2], high=self.domain[3], size=(self.n_observation_pts, 1))
        u = np.sin(x - self.c_exact * t)
        
        return (tf.convert_to_tensor(x, dtype=tf.float32),
                tf.convert_to_tensor(t, dtype=tf.float32),
                tf.convert_to_tensor(u, dtype=tf.float32))

    #  define loss function
    # ---------------------
    @tf.function
    def get_loss(self, 
                 xic, tic, uic, u_t_ic, 
                 xbc, tbc, ubc, 
                 x_col, t_col, f_exact,
                 x_obs, t_obs, u_obs):
        # loss and gradient
        with tf.GradientTape() as tape:
            # initial condition points
            pred_uic = self.compute_u(xic, tic)
            pred_u_t_ic = self.compute_u_t(xic, tic)
            mse_ic = self.loss_fn(uic, pred_uic) + self.loss_fn(u_t_ic, pred_u_t_ic)

            # boundary condition points
            pred_ubc = self.compute_u(xbc, tbc)
            mse_bc = self.loss_fn(ubc, pred_ubc)

            # collocation points
            pred_f = self.pde_residual(x_col, t_col)
            mse_pde = self.loss_fn(f_exact, pred_f)
            
            # observation points
            pred_uobs = self.compute_u(x_obs, t_obs)
            mse_obs = self.loss_fn(u_obs, pred_uobs)

            # combined loss
            mse = mse_ic + mse_bc + mse_pde + mse_obs
            # scale different components of loss
            w1 = 0.125 / (mse_ic/mse)
            w2 = 0.125 / (mse_bc/mse)
            w3 = 0.250 / (mse_pde/mse)
            w4 = 0.500 / (mse_obs/mse)
            loss = w1 * mse_ic + w2 * mse_bc + w3 * mse_pde + w4 * mse_obs
            
        # compute gradients and update variables:
        # 1. get the trainable variables:
        var = self.nn_model.trainable_variables
        # 2. add external variable "c" to training:
        var.append(self.c)
        # 3. compute the gradient:
        grads = tape.gradient(loss, var)
        # 4. apply gradient descent:
        # If you use zip() with n arguments, then the function will 
        # return an iterator that generates tuples of length n.
        self.optimizer.apply_gradients(zip(grads, var))
        
        return loss
    
    def train(self, 
              learning_rate=1e-3, 
              n_iter=100, 
              disp_iter=10, 
              resample_initcond_pts=False, 
              resample_boundary_pts=False, 
              resample_collocation_points=False):
        
        self.optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
        self.loss_fn = tf.keras.losses.MeanSquaredError()
        f_exact = tf.convert_to_tensor(np.zeros((self.n_collocation_pts, 1)))
        loss_history = np.empty(n_iter) 
        
        for it in np.arange(n_iter):
            # initial condition pts
            if it==0 or resample_initcond_pts:
                xic, tic, uic, u_t_ic = self.initial_condition_points()
            # boundary condition pts    
            if it==0 or resample_boundary_pts:
                xbc, tbc, ubc = self.boundary_points()
            # PDE collocation pts    
            if it==0 or resample_collocation_points:
                x_col, t_col = self.collocation_points()
            # observation pts    
            if it==0:
                x_obs, t_obs, u_obs = self.observation_points()
                
            loss = self.get_loss(xic, tic, uic, u_t_ic, 
                                 xbc, tbc, ubc, 
                                 x_col, t_col, f_exact,
                                 x_obs, t_obs, u_obs).numpy()
            loss_history[it] = loss

            if it == 0:
                print('iter,  loss,  estimated wave speed')
            if it % disp_iter == 0:
                print('{:5d} {:f} {:6.03f}'.format(it, loss, self.c.numpy()[0]))

        return loss_history

# visualization
# -------------
def visualize_solution_field(domain, solution, title):
    plt.figure(figsize=(9, 3), dpi=120)
    plt.imshow(solution.T,
               cmap='rainbow',
               interpolation='nearest',
               extent=[domain[2], domain[3], domain[0], domain[1]],
               origin='lower',
               aspect='auto',
               vmin=-1, vmax=1)
    plt.xlabel('t')
    plt.ylabel('x')
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.show()
    return

# problem setup and performance evaluation
# ----------------------------------------
# 1. problem setup
wave_speed = 1 # true value
wave_speed_initial_guess = 1.2 # starting value for optimization
domain = (0, 2 * math.pi, 0, 10)
pinn = PDE_PINN(domain = domain, 
                wave_speed_exact = wave_speed, 
                wave_speed_ig = wave_speed_initial_guess)
pinn.neural_net(nn_arch=[32, 16, 16, 8], input_shape=2)        

# 2. optimization params
n_iter=10000
disp_iter=200
loss_history = pinn.train(n_iter=n_iter, disp_iter=disp_iter)
plt.plot(loss_history)
plt.xlabel('iterations')
plt.ylabel('loss')
plt.show()

# 3. assess quality of the solution
grid_density = 201
xg, tg = np.meshgrid(np.linspace(domain[0], domain[1], grid_density), 
                     np.linspace(domain[2], domain[3], grid_density))
                     
# 3.1. PINN estimation
u_pinn = pinn.compute_u(tf.convert_to_tensor(xg.reshape((-1, 1))), 
                        tf.convert_to_tensor(tg.reshape((-1, 1))))
                   
# 3.2. exact solution
u_exact = np.zeros_like(xg)               
for x_i in range(grid_density):
    for t_i in range(grid_density):
        x = xg[x_i][t_i]
        t = tg[x_i][t_i]
        u_exact[x_i][t_i] = np.sin(x - wave_speed * t)                   

# 3.3. estimation error
error = u_pinn.numpy().reshape(xg.shape) - u_exact

# 4. visualize results
visualize_solution_field(domain, u_pinn.numpy().reshape(xg.shape), 'pinn estimation')
visualize_solution_field(domain, u_exact, 'exact solution')
visualize_solution_field(domain, error, 'estimation_error')