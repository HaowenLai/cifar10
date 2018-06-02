#!/usr/bin/python3

# @author: LHW
# @date  : 2018/2/11

#mport sys
#sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import time

import numpy as np
import tensorflow as tf
#from tensorflow.examples.tutorials.mnist import input_data


# a class containing all the parameters
class Parameter:
    def __init__(self):
        self.learning_rate = 0.0005
        self.max_steps     = 2000
        self.hidden        = (64,32,32)
        self.batch_size    = 1
        self.summary_dir   = '/home/savage/Public/03181059/temp'
        self.log_dir       = '/home/savage/Public/03181059/temp'
        self.restore_file  = '/home/savage/Public/03181059/temp/model.ckpt-999'
        self.fake_data     = False

# a class represents a specific nerual network
class TfGraph():
    def __init__(self):
        # The MNIST dataset has 10 classes, representing the digits 0 through 9.
        # The MNIST images are always 28x28 pixels.
        # The MNIST images are grey scaled(one channel)
        self.NUM_CLASSES    =3
        self.IMAGE_WIDTH    =80
        self.IMAGE_HEIGHT   =60
        self.IMAGE_PIXELS   =self.IMAGE_WIDTH*self.IMAGE_HEIGHT
        self.COLOR          =1

    def inference(self,images, hidden):
        x_image = tf.reshape(images, 
                       [-1,self.IMAGE_HEIGHT,self.IMAGE_WIDTH,self.COLOR])
       
        # conv 1
        with tf.name_scope('conv1'):
            weights=tf.Variable(
                      tf.truncated_normal([5, 5, 1, hidden[0] ],stddev=0.1),
                      name='weights')
            biases = tf.Variable(tf.zeros([hidden[0]]), name='biases')
            h_conv1 = tf.nn.relu(
                    tf.nn.conv2d(x_image,weights,strides=[1, 1, 1, 1], padding='SAME')+
                    biases)
            h_pool1 = tf.nn.max_pool(h_conv1,ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1], padding='SAME')
            
        # conv 2
        with tf.name_scope('conv2'):
            weights=tf.Variable(
                      tf.truncated_normal([7, 7, hidden[0],hidden[1]],stddev=0.1),
                      name='weights')
            biases = tf.Variable(tf.zeros([hidden[1]]), name='biases')
            h_conv2 = tf.nn.relu(
                    tf.nn.conv2d(h_pool1,weights,strides=[1, 1, 1, 1], padding='SAME')+
                    biases)
            h_pool2 = tf.nn.max_pool(h_conv2,ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1], padding='SAME')
        
        # conv 3
        with tf.name_scope('conv3'):
            weights=tf.Variable(
                      tf.truncated_normal([5, 5, hidden[1],hidden[2]],stddev=0.1),
                      name='weights')
            biases = tf.Variable(tf.zeros([hidden[2]]), name='biases')
            h_conv3 = tf.nn.relu(
                    tf.nn.conv2d(h_pool2,weights,strides=[1, 1, 1, 1], padding='SAME')+
                    biases)
            h_pool3 = tf.nn.max_pool(h_conv3,ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1], padding='SAME')
                     
        # dense connected layer
        with tf.name_scope('dense_connected'):
            weights=tf.Variable(
                      tf.truncated_normal([8*10*hidden[2],1024],stddev=0.1),
                      name='weights')
            biases = tf.Variable(tf.zeros([1024]), name='biases')
            h_pool3_flat = tf.reshape(h_pool3,[-1,8*10*hidden[2]])
            h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat,weights)+biases)
        
    
        # Linear
        with tf.name_scope('softmax_linear'):
            weights = tf.Variable(
                        tf.truncated_normal([1024, self.NUM_CLASSES], stddev=0.1),
                        name='weights')
            biases = tf.Variable(tf.zeros([self.NUM_CLASSES]),name='biases')
            logits = tf.matmul(h_fc1, weights) + biases
    
        return logits  # end inference()


    def loss(self,logits, labels):
        labels = tf.to_int64(labels)
        # cross_entropy 交叉熵
        return tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)


    def training(self,loss, learning_rate):
        # Add a scalar summary for the snapshot loss.
        tf.summary.scalar('loss', loss)
        # Create the gradient descent optimizer with the given learning rate.
        optimizer = tf.train.AdamOptimizer(learning_rate)
        # Create a variable to track the global step.
        global_step = tf.Variable(0, name='global_step', trainable=False)
        # Use the optimizer to apply the gradients that minimize the loss
        # (and also increment the global step counter) as a single training step.
        train_op = optimizer.minimize(loss, global_step=global_step)
        return train_op


    def evaluation(self,logits, labels):
        # For a classifier model, we can use the in_top_k Op.
        # It returns a bool tensor with shape [batch_size] that is true for
        # the examples where the label is in the top k (here k=1)
        # of all logits for that example.
        correct = tf.nn.in_top_k(logits, labels, 1)
        # Return the number of true entries.
        return tf.reduce_sum(tf.cast(correct, tf.int32))

class TrainImage:
    def __generate_index_list(self):
        self.index_list = []
        while len(self.index_list) != self.total_number:
            c=np.random.randint(0,self.total_number)
            if c not in self.index_list:
                self.index_list.append(c)
    
    def __init__(self):
        self.img_path     = '/home/savage/Public/03181059/test3/train'
        self.total_number = 600
        self.index_st     = 0
        self.index_end    = 600
        self.label_per100 = ([0],[1],[2],[0],[1],[2])
        # ------------------------------------
        self.index        = 0        
        self.index_list   = []
        
        self.__generate_index_list()
    
  
    def next_image(self):
        #img_index = np.random.randint(self.index_st,self.index_end)
        #img = cv2.imread(self.img_path+'/%d.jpg' % img_index,0)
        img = cv2.imread(self.img_path+'/%d.jpg' % self.index_list[self.index],0)
        img = cv2.resize(img,(80,60))
        img = img * 1.0 /255.0  # convert to float
        img_label = np.array(self.label_per100[self.index_list[self.index]//100])
        
        self.index = self.index + 1
        if self.index == self.index_end:
            self.index = self.index_st
            self.__generate_index_list()
        
        return (img,img_label)


class TestImage:
    def __init__(self):
        self.img_path     = '/home/savage/Public/03181059/test3/test'
        self.total_number = 300
        self.index_st     = 300
        self.index_end    = 600
        self.index        = self.index_st
        self.bias         = 3
        self.label_per100 = ([0],[1],[2])

    def next_image(self):
        #img_index = np.random.randint(self.index_st,self.index_end)
    
        img = cv2.imread(self.img_path+'/%d.jpg' % self.index,0)
        img = cv2.resize(img,(80,60))
        img = img * 1.0 / 255.0  # convert to float
        img_label = np.array(self.label_per100[self.index//100 - self.bias])
        
        self.index = self.index + 1
        if self.index == self.index_end:
            self.index = self.index_st
        
        return (img, img_label)




# Basic model parameters as external flags.
FLAGS = Parameter()
# The specific nerual network object
mnist = TfGraph()
#accuracy record
accuracy_list=[]




def placeholder_inputs(batch_size):
    """Generate placeholder variables to represent the input tensors.

  These placeholders are used as inputs by the rest of the model building
  code and will be fed from the downloaded data in the .run() loop, below.

  Args:
    batch_size: The batch size will be baked into both placeholders.

  Returns:
    images_placeholder: Images placeholder.
    labels_placeholder: Labels placeholder.
    """
    # Note that the shapes of the placeholders match the shapes of the full
    # image and label tensors, except the first dimension is now batch_size
    # rather than the full size of the train or test data sets.
    images_placeholder = tf.placeholder(tf.float32, 
                               shape=(mnist.IMAGE_HEIGHT,mnist.IMAGE_WIDTH),name = 'input')
    labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size))
    return images_placeholder, labels_placeholder


def fill_feed_dict(data_set, images_pl, labels_pl):
    """Fills the feed_dict for training the given step.

  A feed_dict takes the form of:
  feed_dict = {
      <placeholder>: <tensor of values to be passed for placeholder>,
      ....
  }

  Args:
    data_set:  The set of images and labels, from input_data.read_data_sets()
    images_pl: The images placeholder, from placeholder_inputs().
    labels_pl: The labels placeholder, from placeholder_inputs().

  Returns:
    feed_dict: The feed dictionary mapping from placeholders to values.
    """
    # Create the feed_dict for the placeholders filled with the next
    # `batch size` examples.
    images_feed, labels_feed = data_set.next_image()
    feed_dict = {images_pl: images_feed,
                 labels_pl: labels_feed}
    return feed_dict


def do_eval(sess,
            eval_correct,
            images_placeholder,
            labels_placeholder,
            data_set):
 
    # And run one epoch of eval.
    true_count = 0  # Counts the number of correct predictions.
    steps_per_epoch = data_set.total_number // FLAGS.batch_size
    num_examples = steps_per_epoch * FLAGS.batch_size
    for step in range(steps_per_epoch):
        feed_dict = fill_feed_dict(data_set,
                                   images_placeholder,
                                   labels_placeholder)
        true_count += sess.run(eval_correct, feed_dict=feed_dict)
    
    precision = true_count / num_examples
    print('Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
            (num_examples, true_count, precision))

    #add accuracy summary
    accuracy_list.append(precision)
    

"""Train MNIST for a number of steps."""
# Get the sets of images and labels for training, validation, and test on MNIST.
data_sets = [TrainImage(),TestImage()]

# Tell TensorFlow that the model will be built into the default Graph.
#with tf.Graph().as_default():
# Generate placeholders for the images and labels.
images_placeholder, labels_placeholder = placeholder_inputs(FLAGS.batch_size)

# Build a Graph that computes predictions from the inference model.
logits = mnist.inference(images_placeholder, FLAGS.hidden)

# Add to the Graph the Ops for loss calculation.
loss = mnist.loss(logits, labels_placeholder)

# Add to the Graph the Ops that calculate and apply gradients.
train_op = mnist.training(loss, FLAGS.learning_rate)

# Add the Op to compare the logits to the labels during evaluation.
eval_correct = mnist.evaluation(logits, labels_placeholder)

# Add the variable initializer Op.
init = tf.global_variables_initializer()

# Create a saver for writing training checkpoints.
saver = tf.train.Saver()

# Create a session for running Ops on the Graph.
sess = tf.Session()

# Build the summary operation based on the TF collection of Summaries.
summary_op = tf.summary.merge_all()
summary_writer = tf.summary.FileWriter(FLAGS.summary_dir,graph=sess.graph)

# And then after everything is built:
# Run the Op to initialize the variables.
sess.run(init)
start_time = time.time()

# =================================================
# restore parameters from file
#saver.restore(sess, FLAGS.restore_file)
# =================================================

for step in range(FLAGS.max_steps):
    feed_dict = fill_feed_dict(data_sets[0],
                               images_placeholder,
                               labels_placeholder)
    
    _,loss_value = sess.run([train_op, loss], feed_dict=feed_dict)
    
    """
    if step % 20 == 0:
        duration = time.time() - start_time
        start_time = time.time()
        print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
        
        logit_value=sess.run(logits, feed_dict=feed_dict)
        print('logit value is', logit_value,'\n')
    """
    start_time = time.time()
    
    if step % 100 == 0:
        print('\nNo.%d,Test Data Eval:' % step)
        print('time consumed %.3f' % (time.time() - start_time));
        start_time = time.time()
        print('loss: %.4f' % loss_value)
        do_eval(sess,
                eval_correct,
                images_placeholder,
                labels_placeholder,
                data_sets[1])

print('training done !\n')

saver.save(sess, FLAGS.log_dir+'/model.ckpt', global_step=step)

print('Test Data Eval:')
do_eval(sess,
        eval_correct,
        images_placeholder,
        labels_placeholder,
        data_sets[1])

for accu in accuracy_list:
    print(accu)
    
"""
def main(eval_number):
    print('Training Data Eval:')
    start_time = time.time()
    do_eval(sess,
            eval_correct,
            images_placeholder,
            labels_placeholder,
            data_sets.validation)
    
    duration = time.time() - start_time
    print('it takes %.3f sec\n' % duration)

    return (30,0.08)
"""

"""
def main():
    # actually eval_number is always 100, here it is a false value
    start_time = time.time()
    feed_dict = fill_feed_dict(data_sets.train,
                               images_placeholder,
                               labels_placeholder)

    
    true_count = sess.run(eval_correct,feed_dict=feed_dict)

    precision = true_count / eval_number
    print('Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
            (eval_number, true_count, precision))

    duration = time.time() - start_time
    print('it takes %.3f sec\n' % duration)
    
    return (true_count,precision,duration)
"""
