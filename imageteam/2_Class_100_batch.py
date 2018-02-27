import cv2
import os
import SimpleITK as sitk
import numpy as np
import math
import tensorflow as tf
from tflearn.layers.conv import global_avg_pool
from tensorflow.contrib.layers import batch_norm, flatten
from tensorflow.contrib.framework import arg_scope
train_data='CBIS-DDSM/Two_Hist2_CC_cut_3000_X3_train/'
test_data='CBIS-DDSM/Two_Hist2_CC_cut_3000_X3_test/'
train_dir = 'traindata/'
# Hyperparameter
growth_k = 12
nb_block = 4 # how many (dense block + Transition Layer) ?
init_learning_rate = 1e-4
epsilon = 1e-8 # AdamOptimizer epsilon
dropout_rate = 0.2

# Momentum Optimizer will use
nesterov_momentum = 0.9
weight_decay = 1e-4

# Label & batch_size
train_num = 27000
test_num = 2100
class_num = 2
batch_size = 60
total_epochs = 5
def equalizeHist16bit(img):
    hist, bins = np.histogram(img.flatten(), 65536, [0, 65536])
    cdf = hist.cumsum()
    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 65535 / (cdf_m.max() - cdf_m.min())
    cdf = np.ma.filled(cdf_m, 0).astype('uint16')
    return cdf[img]
def conv_layer(input, filter, kernel, stride=1, layer_name="conv"):
    with tf.name_scope(layer_name):
        network = tf.layers.conv2d(inputs=input, filters=filter, kernel_size=kernel, strides=stride, padding='SAME')
        return network

def Global_Average_Pooling(x, stride=1):
    """
    width = np.shape(x)[1]
    height = np.shape(x)[2]
    pool_size = [width, height]
    return tf.layers.average_pooling2d(inputs=x, pool_size=pool_size, strides=stride) # The stride value does not matter
    It is global average pooling without tflearn
    """

    return global_avg_pool(x, name='Global_avg_pooling')
    # But maybe you need to install h5py and curses or not


def Batch_Normalization(x, training, scope):
    with arg_scope([batch_norm],
                   scope=scope,
                   updates_collections=None,
                   decay=0.9,
                   center=True,
                   scale=True,
                   zero_debias_moving_mean=True) :
        return tf.cond(training,
                       lambda : batch_norm(inputs=x, is_training=training, reuse=None),
                       lambda : batch_norm(inputs=x, is_training=training, reuse=True))

def Drop_out(x, rate, training) :
    return tf.layers.dropout(inputs=x, rate=rate, training=training)

def Relu(x):
    return tf.nn.relu(x)

def Average_pooling(x, pool_size=[2,2], stride=2, padding='VALID'):
    return tf.layers.average_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)


def Max_Pooling(x, pool_size=[3,3], stride=2, padding='VALID'):
    return tf.layers.max_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)

def Concatenation(layers) :
    return tf.concat(layers, axis=3)

def Linear(x) :
    return tf.layers.dense(inputs=x, units=class_num, name='linear')



class DenseNet():
    def __init__(self, x, nb_blocks, filters, training):
        self.nb_blocks = nb_blocks
        self.filters = filters
        self.training = training
        self.model = self.Dense_net(x)


    def bottleneck_layer(self, x, scope):
        # print(x)
        with tf.name_scope(scope):
            x = Batch_Normalization(x, training=self.training, scope=scope+'_batch1')
            x = Relu(x)
            x = conv_layer(x, filter=4 * self.filters, kernel=[1,1], layer_name=scope+'_conv1')
            x = Drop_out(x, rate=dropout_rate, training=self.training)

            x = Batch_Normalization(x, training=self.training, scope=scope+'_batch2')
            x = Relu(x)
            x = conv_layer(x, filter=self.filters, kernel=[3,3], layer_name=scope+'_conv2')
            x = Drop_out(x, rate=dropout_rate, training=self.training)

            # print(x)

            return x

    def transition_layer(self, x, scope):
        with tf.name_scope(scope):
            x = Batch_Normalization(x, training=self.training, scope=scope+'_batch1')
            x = Relu(x)
            x = conv_layer(x, filter=self.filters, kernel=[1,1], layer_name=scope+'_conv1')
            x = Drop_out(x, rate=dropout_rate, training=self.training)
            x = Average_pooling(x, pool_size=[2,2], stride=2)

            return x

    def dense_block(self, input_x, nb_layers, layer_name):
        with tf.name_scope(layer_name):
            layers_concat = list()
            layers_concat.append(input_x)

            x = self.bottleneck_layer(input_x, scope=layer_name + '_bottleN_' + str(0))

            layers_concat.append(x)

            for i in range(nb_layers - 1):
                x = Concatenation(layers_concat)
                x = self.bottleneck_layer(x, scope=layer_name + '_bottleN_' + str(i + 1))
                layers_concat.append(x)
	    
	    x = Concatenation(layers_concat)

            return x

    def Dense_net(self, input_x):
        x = conv_layer(input_x, filter=2 * self.filters, kernel=[7,7], stride=2, layer_name='conv0')
        x = Max_Pooling(x, pool_size=[3,3], stride=2)


        '''
        for i in range(self.nb_blocks) :
            # 6 -> 12 -> 48
            x = self.dense_block(input_x=x, nb_layers=4, layer_name='dense_'+str(i))
            x = self.transition_layer(x, scope='trans_'+str(i))
        '''

        x = self.dense_block(input_x=x, nb_layers=6, layer_name='dense_1')
        x = self.transition_layer(x, scope='trans_1')

        x = self.dense_block(input_x=x, nb_layers=12, layer_name='dense_2')
        x = self.transition_layer(x, scope='trans_2')

        x = self.dense_block(input_x=x, nb_layers=24, layer_name='dense_3')
        x = self.transition_layer(x, scope='trans_3')


        x = self.dense_block(input_x=x, nb_layers=16, layer_name='dense_final')


        # 100 Layer
        x = Batch_Normalization(x, training=self.training, scope='linear_batch')
        x = Relu(x)
        x = Global_Average_Pooling(x)
        x = flatten(x)
        x = Linear(x)


        # x = tf.reshape(x, [-1, 10])
        return x

#224*224=50176
x = tf.placeholder(tf.float32, shape=[None, 50176])
batch_images = tf.reshape(x, [-1, 224, 224, 1])

label = tf.placeholder(tf.float32, shape=[None, 2])

training_flag = tf.placeholder(tf.bool)


learning_rate = tf.placeholder(tf.float32, name='learning_rate')

logits = DenseNet(x=batch_images, nb_blocks=nb_block, filters=growth_k, training=training_flag).model
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=logits))

"""
l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=nesterov_momentum, use_nesterov=True)
train = optimizer.minimize(cost + l2_loss * weight_decay)
In paper, use MomentumOptimizer
init_learning_rate = 0.1
but, I'll use AdamOptimizer
"""

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=epsilon)
train = optimizer.minimize(cost)


correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(label, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

loss_summary=tf.summary.scalar('loss', cost)
training_summary=tf.summary.scalar('Training Accuracy', accuracy)
testing_summary=tf.summary.scalar('Testing Accuracy', accuracy)
saver = tf.train.Saver(tf.global_variables())

with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state('./model/dense_2.ckpt')
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())

    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter('./logs_2', sess.graph)
    train_image= []
    train_label= []
    test_image = []
    test_label = []
    files=[]
    files_t=[]
    num_input_train=0
    for filename in os.listdir(train_data):
        if num_input_train >=train_num:
            break
        num_input_train += 1
        files.append(filename)

    for old_index in range(len(files)):
        new_index = np.random.randint(old_index + 1)
        files[old_index], files[new_index] = files[new_index], files[old_index]
    
    for l in files:
        img = cv2.imread(train_data+l,-1)
        train_image.append(img.flatten().astype(np.float32))
        a=[0,0]
        if int(l[1]) ==1:
            a[0]=1
        elif int(l[1]) ==2:
            a[1]=1
        else:
            print("error")
        train_label.append(a)
	print('loading training data: '+l)
	"""
	img = img.reshape(-1)
	img[img < 10000]=65535
	img[img > 60000]=65535
        img = np.delete(img, np.where(img ==65535), axis=0)
        b = b.reshape(-1)
	b[b < 10000]=65535
	b[b > 60000]=65535
	b = np.delete(b, np.where(b ==65535), axis=0)
	c = c.reshape(-1)
        c[c < 10000]=65535
        c[c > 60000]=65535
        c = np.delete(c, np.where(c ==65535), axis=0)
	d = d.reshape(-1)
        d[d < 10000]=65535
        d[d > 60000]=65535
        d = np.delete(d, np.where(d ==65535), axis=0)
	"""
    num_input_test=0
    for filename_t in os.listdir(test_data):
        if num_input_test >=test_num:
            break
        num_input_test += 1
        files_t.append(filename_t)

    for old_index in range(len(files_t)):
        new_index = np.random.randint(old_index + 1)
        files_t[old_index], files_t[new_index] = files_t[new_index], files_t[old_index]
    
    for l in files_t:
        img = cv2.imread(test_data+l,-1)
	test_image.append(img.flatten().astype(np.float32))
	#print(test_image[:2])
        a=[0,0]
        if int(l[1]) ==1:
            a[0]=1
        elif int(l[1]) ==2:
            a[1]=1
        else:
            print("error")
        test_label.append(a)
	print('loading testing data: '+l)
	"""
	img = img.reshape(-1)
        img[img < 10000]=65535
        img[img > 60000]=65535
        img = np.delete(img, np.where(img ==65535), axis=0)
        b = b.reshape(-1)
        b[b < 10000]=65535
        b[b > 60000]=65535
        b = np.delete(b, np.where(b ==65535), axis=0)
        c = c.reshape(-1)
        c[c < 10000]=65535
        c[c > 60000]=65535
        c = np.delete(c, np.where(c ==65535), axis=0)
        d = d.reshape(-1)
        d[d < 10000]=65535
        d[d > 60000]=65535
        d = np.delete(d, np.where(d ==65535), axis=0)
	"""
    global_step = 0
    epoch_learning_rate = init_learning_rate
    for epoch in range(total_epochs):
        if epoch == (total_epochs * 0.5) or epoch == (total_epochs * 0.75):
            epoch_learning_rate = epoch_learning_rate / 10
        
        total_batch = (int(len(train_image) / batch_size))+1
        for step in range(total_batch):
            batch = batch_size * step
            if batch == len(train_image):
                break

            train_feed_dict = {
                x: train_image[batch:batch + batch_size],
                label: train_label[batch:batch + batch_size],
                learning_rate: epoch_learning_rate,
                training_flag : True
            }

            _, loss,loss_summ = sess.run([train, cost,loss_summary], feed_dict=train_feed_dict)
	    writer.add_summary(loss_summ,global_step=epoch)
            if step % 100 == 0:
                global_step += 100
                train_summary, train_accuracy = sess.run([training_summary, accuracy], feed_dict=train_feed_dict)
                # accuracy.eval(feed_dict=feed_dict)
                print("Step:", step, "Loss:", loss, "Training accuracy:", train_accuracy)
                writer.add_summary(train_summary, global_step=epoch)
       
	count=0
	test_accuracy=0
        for i in range(int(len(test_image) / batch_size)+1):
            #print('i: ',i)
	    batch = batch_size * i
            if batch == len(test_image):
                break
            test_feed_dict = {
                x: test_image[batch:batch + batch_size],
                label: test_label[batch:batch + batch_size],
                learning_rate: epoch_learning_rate,
                training_flag : False
            }
            test_summary, accuracy_rates = sess.run([testing_summary,accuracy], feed_dict=test_feed_dict)
	   # print(accuracy_rates)
	    test_accuracy += accuracy_rates
	    count += 1
	    answer=sess.run(tf.argmax(label,1),feed_dict=test_feed_dict)
            predit=sess.run(tf.argmax(logits, 1), feed_dict=test_feed_dict)
            #print(answer)
	    #print(predit)
            for j in range(len(answer)):
                if answer[j]!=predit[j]:
                    print(files_t[batch+j],predit[j]+1, 'wrong')

        print('Epoch:', '%04d' % (epoch + 1), '/ Accuracy =', test_accuracy/count)
        writer.add_summary(test_summary, global_step=epoch)
    saver.save(sess=sess, save_path='./model/dense_2.ckpt')
