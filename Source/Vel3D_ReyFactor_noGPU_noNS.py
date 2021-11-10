"""
@author: Maziar Raissi
"""

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import numpy as np
import scipy.io
import time
import sys
from itertools import zip_longest

from utilities import neural_net, Navier_Stokes_3D_2, \
                      tf_session, mean_squared_error, relative_error
    
class HFM(object):
    # notational conventions
    # _tf: placeholders for input/output data and points used to regress the equations
    # _pred: output of neural network
    # _eqns: points used to regress the equations
    # _data: input-output data
    # _star: preditions
    
    def __init__(self, t_data, x_data, y_data, z_data, u_data, v_data, w_data,
                    #    t_eqns, x_eqns, y_eqns, z_eqns,
                       layers, batch_size,
                       Rey):
        
        # specs
        self.layers = layers
        self.batch_size = batch_size
        
        # flow properties
        self.Rey = Rey
        
        # data
        [self.t_data, self.x_data, self.y_data, self.z_data, self.u_data, self.v_data, self.w_data] = [t_data, x_data, y_data, z_data, u_data, v_data, w_data]
        # [self.t_eqns, self.x_eqns, self.y_eqns, self.z_eqns] = [t_eqns, x_eqns, y_eqns, z_eqns]
        
        # placeholders
        [self.t_data_tf, self.x_data_tf, self.y_data_tf, self.z_data_tf, self.u_data_tf, self.v_data_tf, self.w_data_tf] = [tf.placeholder(tf.float32, shape=[None, 1]) for _ in range(7)]
        # [self.t_eqns_tf, self.x_eqns_tf, self.y_eqns_tf, self.z_eqns_tf] = [tf.placeholder(tf.float32, shape=[None, 1]) for _ in range(4)]
        
        # physics "uninformed" neural networks
        self.net_cuvwp = neural_net(self.t_data, self.x_data, self.y_data, self.z_data, layers = self.layers)
        
        [self.u_data_pred,
         self.v_data_pred,
         self.w_data_pred,
         self.p_data_pred] = self.net_cuvwp(self.t_data_tf,
                                            self.x_data_tf,
                                            self.y_data_tf,
                                            self.z_data_tf)
         
        # physics "informed" neural networks
        # [self.u_eqns_pred,
        #  self.v_eqns_pred,
        #  self.w_eqns_pred,
        #  self.p_eqns_pred] = self.net_cuvwp(self.t_eqns_tf,
        #                                     self.x_eqns_tf,
        #                                     self.y_eqns_tf,
        #                                     self.z_eqns_tf)
        
        # [self.e1_eqns_pred,
        #  self.e2_eqns_pred,
        #  self.e3_eqns_pred,
        #  self.e4_eqns_pred] = Navier_Stokes_3D_2(self.u_eqns_pred,
        #                                        self.v_eqns_pred,
        #                                        self.w_eqns_pred,
        #                                        self.p_eqns_pred,
        #                                        self.t_eqns_tf,
        #                                        self.x_eqns_tf,
        #                                        self.y_eqns_tf,
        #                                        self.z_eqns_tf,
        #                                        self.Rey)
        
        # loss
        self.loss = 100 * (mean_squared_error(self.u_data_pred, self.u_data_tf) + \
                    mean_squared_error(self.v_data_pred, self.v_data_tf) + \
                    mean_squared_error(self.w_data_pred, self.w_data_tf) ) # + \
                    # 10 * (mean_squared_error(self.e1_eqns_pred, 0.0) + \
                    # mean_squared_error(self.e2_eqns_pred, 0.0) + \
                    # mean_squared_error(self.e3_eqns_pred, 0.0) + \
                    # mean_squared_error(self.e4_eqns_pred, 0.0)) 
        
        # optimizers
        self.learning_rate = tf.placeholder(tf.float32, shape=[])
        self.optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
        self.train_op = self.optimizer.minimize(self.loss)
        
        self.sess = tf_session()
        
        self.saver = tf.train.Saver()
        self.save_path = ''
    
    def train(self, total_time, learning_rate):
        
        N_data = self.t_data.shape[0]
        # N_eqns = self.t_eqns.shape[0]
        
        start_time = time.time()
        running_time = 0
        it = 0
        while running_time < total_time:
            
            idx_data = np.random.choice(N_data, self.batch_size)
            # idx_eqns = np.random.choice(N_eqns, self.batch_size)
            
            (t_data_batch,
             x_data_batch,
             y_data_batch,
             z_data_batch,
             u_data_batch,
             v_data_batch,
             w_data_batch) = (self.t_data[idx_data,:],
                              self.x_data[idx_data,:],
                              self.y_data[idx_data,:],
                              self.z_data[idx_data,:],
                              self.u_data[idx_data,:],
                              self.v_data[idx_data,:],
                              self.w_data[idx_data,:])

            # (t_eqns_batch,
            #  x_eqns_batch,
            #  y_eqns_batch,
            #  z_eqns_batch) = (self.t_eqns[idx_eqns,:],
            #                   self.x_eqns[idx_eqns,:],
            #                   self.y_eqns[idx_eqns,:],
            #                   self.z_eqns[idx_eqns,:])

            tf_dict = {self.t_data_tf: t_data_batch,
                       self.x_data_tf: x_data_batch,
                       self.y_data_tf: y_data_batch,
                       self.z_data_tf: z_data_batch,
                       self.u_data_tf: u_data_batch,
                       self.v_data_tf: v_data_batch,
                       self.w_data_tf: w_data_batch,
                    #    self.t_eqns_tf: t_eqns_batch,
                    #    self.x_eqns_tf: x_eqns_batch,
                    #    self.y_eqns_tf: y_eqns_batch,
                    #    self.z_eqns_tf: z_eqns_batch,
                       self.learning_rate: learning_rate}
            
            self.sess.run([self.train_op], tf_dict)

            
            # Print
            if it % 10 == 0:
                elapsed = time.time() - start_time
                running_time += elapsed/3600.0
                [loss_value,
                 learning_rate_value] = self.sess.run([self.loss,
                                                       self.learning_rate], tf_dict)
                print('It: %d, Loss: %.3e, Time: %.2fs, Running Time: %.2fh, Learning Rate: %.1e'
                      %(it, loss_value, elapsed, running_time, learning_rate_value))
                sys.stdout.flush()
                start_time = time.time()

            if it % 10000 == 0:
                self.saver.save(self.sess, self.save_path)

            it += 1
    
    def predict(self, t_star, x_star, y_star, z_star):
        
        tf_dict = {self.t_data_tf: t_star, self.x_data_tf: x_star, self.y_data_tf: y_star, self.z_data_tf: z_star}
        
        u_star = self.sess.run(self.u_data_pred, tf_dict)
        v_star = self.sess.run(self.v_data_pred, tf_dict)
        w_star = self.sess.run(self.w_data_pred, tf_dict)
        p_star = self.sess.run(self.p_data_pred, tf_dict)
        
        return u_star, v_star, w_star, p_star

    def set_save_path(self, save_path):
        self.save_path = save_path




if __name__ == "__main__":
    layers = [4]
    print(f"Arguments count: {len(sys.argv)}")
    if len(sys.argv) < 5 or len(sys.argv) % 2 == 0:
        print("Wrong arguments!")
        quit()
    save_name  = sys.argv[1]
    Rey_factor = int(sys.argv[2])
    print(f"save_name: {save_name}, Rey_factor: {Rey_factor}")
    args = zip_longest(*[iter(sys.argv[3:])]*2)
    i = 1
    for num, size in args:
        print(f"Group {i+1:>6}: {num, size}")
        layers += int(num)*[int(size)]
        i += 1
    
    layers += [4]

    batch_size = 10000
    
    # layers = [4] + 25*[2*50] + [4]  # output: u v w p
    print(len(layers), layers)
    
    data_path = '/home/rcheng15/scr4_rni2/HFM/Quarter_higher_resolution/'
    safe_path = '/home/rcheng15/HFM/HFM_out/'
    # Load Data
    data = scipy.io.loadmat(data_path + 'velfield.mat')

    Rey = data['Re'] * Rey_factor  # 2, 1.5
    
    # grids for measured points
    t_star = data['t_star'] # T x 1
    x_star = data['x_star'] # N x 1
    y_star = data['y_star'] # N x 1
    z_star = data['z_star'] # N x 1

    # grids for NS equation
    x = data['X'] # N x 1
    y = data['Y'] # N x 1
    z = data['Z'] # N x 1
    
    T = t_star.shape[0]
    N = x_star.shape[0]
    M = x.shape[0]
    
    U_star = data['u'] # N x T
    V_star = data['v'] # N x T
    W_star = data['w'] # N x T
    # P_star = data['P_star'] # N x T
    # C_star = data['C_star'] # N x T

    # Rearrange Data 
    T_star = np.tile(t_star, (1,N)).T # N x T
    X_star = np.tile(x_star, (1,T)) # N x T
    Y_star = np.tile(y_star, (1,T)) # N x T
    Z_star = np.tile(z_star, (1,T)) # N x T
    # T_eqn_star = np.tile(t_star, (1,M)).T
    # X_eqn_star = np.tile(x, (1,T)) # M x T
    # Y_eqn_star = np.tile(y, (1,T)) # M x T
    # Z_eqn_star = np.tile(z, (1,T)) # M x T

        
    ######################################################################
    ######################## Training Data ###############################
    ######################################################################
    
    T_data = T
    N_data = N
    idx_t = np.concatenate([np.array([0]), np.random.choice(T-2, T_data-2, replace=False)+1, np.array([T-1])] )
    idx_x = np.random.choice(N, N_data, replace=False)
    t_data = T_star[:, idx_t][idx_x,:].flatten()[:,None]
    x_data = X_star[:, idx_t][idx_x,:].flatten()[:,None]
    y_data = Y_star[:, idx_t][idx_x,:].flatten()[:,None]
    z_data = Z_star[:, idx_t][idx_x,:].flatten()[:,None]
    u_data = U_star[:, idx_t][idx_x,:].flatten()[:,None]
    v_data = V_star[:, idx_t][idx_x,:].flatten()[:,None]
    w_data = W_star[:, idx_t][idx_x,:].flatten()[:,None]

    #M = tracks.shape[0]
    #idx_x = np.random.choice(M, M, replace=False)
    #t_data = np.array(tracks[idx_x, 3]).flatten()[:,None]
    #x_data = np.array(tracks[idx_x, 0]).flatten()[:,None]
    #y_data = np.array(tracks[idx_x, 1]).flatten()[:,None]
    #z_data = np.array(tracks[idx_x, 2]).flatten()[:,None]
    #u_data = np.array(tracks[idx_x, 5]).flatten()[:,None]
    #v_data = np.array(tracks[idx_x, 6]).flatten()[:,None]
    #w_data = np.array(tracks[idx_x, 7]).flatten()[:,None]

    
    # T_eqns = T
    # M_eqns = M
    # idx_t = np.concatenate([np.array([0]), np.random.choice(T-2, T_eqns-2, replace=False)+1, np.array([T-1])] )
    # idx_x = np.random.choice(M, M_eqns, replace=False)
    # t_eqns = T_eqn_star[:, idx_t][idx_x,:].flatten()[:,None]
    # x_eqns = X_eqn_star[:, idx_t][idx_x,:].flatten()[:,None]
    # y_eqns = Y_eqn_star[:, idx_t][idx_x,:].flatten()[:,None]
    # z_eqns = Z_eqn_star[:, idx_t][idx_x,:].flatten()[:,None]
        
    # Training
    model = HFM(t_data, x_data, y_data, z_data, u_data, v_data, w_data,
                # t_eqns, x_eqns, y_eqns, z_eqns,
                layers, batch_size,
                Rey)
    
    model.set_save_path(safe_path + f'{save_name}_model.ckpt')
    model.train(total_time = 30, learning_rate=1e-3)

    # save model 
    
    # load model
    #sess = tf.Session()
    #saver = tf.train.import_meta_graph(safe_path + 'model.ckpt.meta')
    #saver.restore(sess, tf.train.latest_checkpoint(safe_path))
    #model_save = HFM(t_data, x_data, y_data, z_data, c_data,
    #            t_eqns, x_eqns, y_eqns, z_eqns,
    #            layers, batch_size,
    #            Pec = 100, Rey = 100)
    #model.saver = tf.train.import_meta_graph(safe_path + 'model.ckpt.meta')
    #model.saver.restore(model.sess, tf.train.latest_checkpoint(safe_path))

    # Test Data
    snap = np.array([100])
    t_test = T_star[:,snap]
    x_test = X_star[:,snap]
    y_test = Y_star[:,snap]
    z_test = Z_star[:,snap]
    
    u_test = U_star[:,snap]
    v_test = V_star[:,snap]
    w_test = W_star[:,snap]
    
    # Prediction
    u_pred, v_pred, w_pred, p_pred = model.predict(t_test, x_test, y_test, z_test)
    
    # Error
    error_u = relative_error(u_pred, u_test)
    error_v = relative_error(v_pred, v_test)
    error_w = relative_error(w_pred, w_test)

    print('Error u: %e' % (error_u))
    print('Error v: %e' % (error_v))
    print('Error w: %e' % (error_w))
        
    ################# Save Data ###########################
    

    U_pred = 0*U_star
    V_pred = 0*V_star
    W_pred = 0*W_star
    for snap in range(0,t_star.shape[0]):
        t_test = T_star[:,snap:snap+1]
        x_test = X_star[:,snap:snap+1]
        y_test = Y_star[:,snap:snap+1]
        z_test = Z_star[:,snap:snap+1]
        
        u_test = U_star[:,snap:snap+1]
        v_test = V_star[:,snap:snap+1]
        w_test = W_star[:,snap:snap+1]

    
        # Prediction
        u_pred, v_pred, w_pred, p_pred = model.predict(t_test, x_test, y_test, z_test)
        
        U_pred[:,snap:snap+1] = u_pred
        V_pred[:,snap:snap+1] = v_pred
        W_pred[:,snap:snap+1] = w_pred

    
        ## Error
        error_u = relative_error(u_pred, u_test)
        error_v = relative_error(v_pred, v_test)
        error_w = relative_error(w_pred, w_test)

        print('Error u: %e' % (error_u))
        print('Error v: %e' % (error_v))
        print('Error w: %e' % (error_w))

    
    scipy.io.savemat( safe_path + f"Vel3D_{save_name}_results_{time.strftime('%d_%m_%Y')}.mat",
                     {'U_pred':U_pred, 'V_pred':V_pred, 'W_pred':W_pred})
