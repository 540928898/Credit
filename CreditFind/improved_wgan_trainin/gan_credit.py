import os, sys
sys.path.append(os.getcwd())

import random
import time

#import matplotlib
#matplotlib.use('Agg')
#import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import sklearn.datasets

import tflib as lib
import tflib.ops.linear2 as linear
#import tflib.plot

MODE = 'wgan-gp' # wgan or wgan-gp
DATASET = '8gaussians' # 8gaussians, 25gaussians, swissroll
DIM = 256 # Model dimensionality
FIXED_GENERATOR = False # whether to hold the generator fixed at real data plus
                        # Gaussian noise, as in the plots in the paper
LAMBDA = .1 # Smaller lambda makes things faster for toy tasks, but isn't
            # necessary if you increase CRITIC_ITERS enough
CRITIC_ITERS = 5 # How many critic iterations per generator iteration
BATCH_SIZE = 2 # Batch size
ITERS = 10000 # how many generator iterations to train for
fakenum=1000
lib.print_model_settings(locals().copy())

def ReLULayer(name, n_in, n_out, inputs):
    output = lib.ops.linear2.Linear(
        name+'.Linear',
        n_in,
        n_out,
        inputs,
        initialization='he'
    )
    output = tf.nn.relu(output)
    return output

def Generator(n_samples, real_data):
    if FIXED_GENERATOR:
        return real_data + (1.*tf.random_normal(tf.shape(real_data)))
    else:
        noise = tf.random_normal([n_samples, 29])
        output = ReLULayer('Generator.1', 29, DIM, noise)
        output = ReLULayer('Generator.2', DIM, DIM, output)
        output = ReLULayer('Generator.3', DIM, DIM, output)
        output = lib.ops.linear2.Linear('Generator.4', DIM, 29, output)
        return output

def Discriminator(inputs):
    output = ReLULayer('Discriminator.1', 29, DIM, inputs)
    output = ReLULayer('Discriminator.2', DIM, DIM, output)
    output = ReLULayer('Discriminator.3', DIM, DIM, output)
    output = lib.ops.linear2.Linear('Discriminator.4', DIM, 1, output)
    return tf.reshape(output, [-1])

real_data = tf.placeholder(tf.float32, shape=[None, 29])
fake_data = Generator(BATCH_SIZE, real_data)

disc_real = Discriminator(real_data)
disc_fake = Discriminator(fake_data)

# WGAN loss
disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)
#求均值
gen_cost = -tf.reduce_mean(disc_fake)

# WGAN gradient penalty
if MODE == 'wgan-gp':
    alpha = tf.random_uniform(
        shape=[BATCH_SIZE,1], 
        minval=0.,
        maxval=1.
    )
    #加权
    interpolates = alpha*real_data + ((1-alpha)*fake_data)
    
    disc_interpolates = Discriminator(interpolates)
    #求梯度 
    gradients = tf.gradients(disc_interpolates, [interpolates])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
    gradient_penalty = tf.reduce_mean((slopes-1)**2)
 
    disc_cost += LAMBDA*gradient_penalty

disc_params = lib.params_with_name('Discriminator')
gen_params = lib.params_with_name('Generator')

if MODE == 'wgan-gp':
    disc_train_op = tf.train.AdamOptimizer(
        learning_rate=1e-4, 
        beta1=0.5, 
        beta2=0.9
    ).minimize(
        disc_cost, 
        var_list=disc_params
    )
    if len(gen_params) > 0:
        gen_train_op = tf.train.AdamOptimizer(
            learning_rate=1e-4, 
            beta1=0.5, 
            beta2=0.9
        ).minimize(
            gen_cost, 
            var_list=gen_params
        )
    else:
        gen_train_op = tf.no_op()

else:
    disc_train_op = tf.train.RMSPropOptimizer(learning_rate=5e-5).minimize(
        disc_cost, 
        var_list=disc_params
    )
    if len(gen_params) > 0:
        gen_train_op = tf.train.RMSPropOptimizer(learning_rate=5e-5).minimize(
            gen_cost, 
            var_list=gen_params
        )
    else:
        gen_train_op = tf.no_op()


    # Build an op to do the weight clipping
    clip_ops = []
    for var in disc_params:
        clip_bounds = [-.01, .01]
        clip_ops.append(
            tf.assign(
                var, 
                tf.clip_by_value(var, clip_bounds[0], clip_bounds[1])
            )
        )
    clip_disc_weights = tf.group(*clip_ops)

print( "Generator params:")
for var in lib.params_with_name('Generator'):
    print( "\t{}\t{}".format(var.name, var.get_shape()))
print ("Discriminator params:")
for var in lib.params_with_name('Discriminator'):
    print ("\t{}\t{}".format(var.name, var.get_shape()))

frame_index = [0]
def generate_image(true_dist):
    """
    Generates and saves a plot of the true distribution, the generator, and the
    critic.
    """
    N_POINTS = 128
    RANGE = 3

    points = np.zeros((N_POINTS, N_POINTS, 2), dtype='float32')
    points[:,:,0] = np.linspace(-RANGE, RANGE, N_POINTS)[:,None]
    points[:,:,1] = np.linspace(-RANGE, RANGE, N_POINTS)[None,:]
    points = points.reshape((-1,2))
    samples, disc_map = session.run(
        [fake_data, disc_real], 
        feed_dict={real_data:points}
    )
    #一般来说 需要执行两遍 
    disc_map = session.run(disc_real, feed_dict={real_data:points})

    plt.clf()

    x = y = np.linspace(-RANGE, RANGE, N_POINTS)
    plt.contour(x,y,disc_map.reshape((len(x), len(y))).transpose())

    plt.scatter(true_dist[:, 0], true_dist[:, 1], c='orange',  marker='+')
    plt.scatter(samples[:, 0],    samples[:, 1],    c='green', marker='+')

    plt.savefig('frame'+str(frame_index[0])+'.pdf')
    frame_index[0] += 1

# Dataset iterator
    

def inf_train_gen():
    if DATASET == '25gaussians':
        dataset = []
        for i in range(100000/25):
            for x in range(-2, 3):
                for y in range(-2, 3):
                    point = np.random.randn(2)*0.05
                    point[0] += 2*x
                    point[1] += 2*y
                    dataset.append(point)
        dataset = np.array(dataset, dtype='float32')
        np.random.shuffle(dataset)
        dataset /= 2.828 # stdev
        while True:
            for i in range(len(dataset)/BATCH_SIZE):
                yield dataset[i*BATCH_SIZE:(i+1)*BATCH_SIZE]

    elif DATASET == 'swissroll':

        while True:
            data = sklearn.datasets.make_swiss_roll(
                n_samples=BATCH_SIZE, 
                noise=0.25
            )[0]
            data = data.astype('float32')[:, [0, 2]]
            data /= 7.5 # stdev plus a little
            yield data

    elif DATASET == '8gaussians':
        scale = 2.
        centers=[ x for x in range(29)] 
#        centers = [
#            (1,0),
#            (-1,0),
#            (0,1),
#            (0,-1),
#            (1./np.sqrt(2), 1./np.sqrt(2)),
#            (1./np.sqrt(2), -1./np.sqrt(2)),
#            (-1./np.sqrt(2), 1./np.sqrt(2)),
#            (-1./np.sqrt(2), -1./np.sqrt(2))
#        ]
#        centers = [(scale*x,scale*y) for x,y in centers]
        while True:
            dataset = []
            for i in range(BATCH_SIZE):
                point = np.random.randn(29)*.02
                for enum,evalue in enumerate(centers):
                    point[enum]+=evalue                                                   
                dataset.append(point)
            dataset = np.array(dataset, dtype='float32')
            dataset /= 1.414 # stdev
            yield dataset
            
def fit(datafunc,x_train,y_train):
    fakedata=[]
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        gen = datafunc(x_train,y_train)
        #ITER 1000
        for iteration in range(ITERS):
    #        # Train generator
            if iteration > 0:
                _ = session.run(gen_train_op)
    #        # Train critic
            for i in range(CRITIC_ITERS):
                _data = gen.__next__()
                _disc_cost, _ = session.run(
                    [disc_cost, disc_train_op],
                    feed_dict={real_data: _data}
                )
            if iteration %100 == 99:
                print("iteration: "+str(iteration))
                
#                if MODE == 'wgan':
#                    _ = session.run([clip_disc_weights])
            # Write logs and save samples 
        for inter in range(fakenum):
            samples, disc_map = session.run(
                    [fake_data, disc_real], 
                    feed_dict={real_data:_data})
            fakedata.extend(samples[1].reshape(1,29))
        fakelabel=np.ones((fakenum,1))
        print("you are full?")
        return fakedata,fakelabel
        
        
        
        
        
        
        
        
        
        