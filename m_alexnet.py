import tensorflow as tf

from dataset_loader import load_dataset


batch_size = 128
learning_rate = 0.01
decay = 5e-4
momentum = 0.9
input_size = 3468
num_of_classes = 50
kernel_std_norm = 0.01



def add_to_tboard(in_tensor, name):
    with tf.name_scope(name):
        mean = tf.reduce_mean(in_tensor)
        tf.summary.scalar('Mean', mean)
        with tf.name.scope('sd'):
            sd = tf.sqrt(tf.reduce_mean(tf.square(in_tensor - mean)))
        tf.summary.scalar('Standard Deviation', sd)
        tf.summary.scalar('Minimum', tf.reduce_min(in_tensor))
        tf.summary.scalar('Maximum', tf.reduce_max(in_tensor))
        tf.summary.histogram('Histogram', in_tensor)

def conv1(in_tensor):
    kernel = tf.Variable(tf.truncated_normal([3,3,3,96],dtype=tf.float32,
                                             stddev=0.01))
    b = tf.Variable(tf.constant(0, shape=[96], dtype=tf.float32),
                    trainable=True)

    conv_layer = tf.nn.conv2d(in_tensor,
                                  kernel,
                                  [1,1,1,1])
    bias = tf.nn.bias_add(conv_layer,b)
    lrn = tf.nn.local_response_normalization(bias,depth_radius=5,bias=2,alpha=1e-4,beta=0.75)
    activation = tf.nn.relu(lrn)
    pooling = tf.nn.max_pool(activation, ksize=[1,2,2,1],
                             strides=[1,1,1,1])