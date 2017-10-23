import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
x_valid_data, y_valid_data = mnist.validation.next_batch(5000)
x_valid_data = x_valid_data.reshape([-1,28,28,1])
x_test_data, y_test_data = mnist.test.next_batch(10000)
x_test_data = x_test_data.reshape([-1,28,28,1])

# #### Convolutional Neural Network

def flat(x_tensor):
    input_shape = x_tensor.get_shape().as_list()
    flatness = tf.reshape(x_tensor,[-1,input_shape[3]*input_shape[2]*input_shape[1]])
    return flatness

def cnn_model(features,labels):
    with tf.name_scope("Input_Layer"):
        input_layer = tf.reshape(features, [-1, 28, 28, 1])
    with tf.name_scope("CONV_Layer_1"):
        conv1 = tf.layers.conv2d(inputs=input_layer,filters=64,kernel_size=[3,3],strides=[1,1], padding='VALID',activation=tf.nn.relu,name='CONV1')
    with tf.name_scope("CONV_Layer_2"):
        conv2 = tf.layers.conv2d(conv1,64,kernel_size=[3,3],strides=[1,1], padding='VALID',activation=tf.nn.relu,name='CONV2')
    with tf.name_scope("CONV_Layer_3"):
        conv3 = tf.layers.conv2d(conv2,64,kernel_size=[3,3],strides=[1,1], padding='VALID',activation=tf.nn.relu,name='CONV3')
    with tf.name_scope("FC_Layer_1"):
        fc1 = tf.layers.dense(flat(conv3),units=512,activation=tf.nn.relu,name='FC1')
    with tf.name_scope("FC_Layer_2"):
        out_layer = tf.layers.dense(fc1, units=10,name='FC2')
    return out_layer

def print_stats(session, feature_batch, label_batch, cost, accuracy):
    accuracy_value = session.run(accuracy, feed_dict={
    x: np.reshape(x_valid_data,[-1,28,28,1]),
    y: y_valid_data})

    loss_value = session.run(cost, feed_dict={
    x: feature_batch,
    y: label_batch})
    print("Loss: {:4.4f} Accuracy: {:4.4f}".format(loss_value, accuracy_value))

#save_model_path = './'
with tf.name_scope('Inputs'):
    x = tf.placeholder(tf.float32,shape=(None,28,28,1),name='x')

with tf.name_scope('Targets'):
    y = tf.placeholder(tf.float32,shape=(None,10),name='y')

with tf.name_scope('Output'):
    logits = cnn_model(x,y)

with tf.name_scope('training'):
    loss = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=logits)
    cost = tf.reduce_mean(loss)
    optimizer = tf.train.AdamOptimizer().minimize(cost)

with tf.name_scope('accuracy'):
    correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')
tf.summary.scalar('accuracy', accuracy)

merged = tf.summary.merge_all()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    train_writer = tf.summary.FileWriter('./train',sess.graph) 
    test_writer = tf.summary.FileWriter('./test') 
    for batch_i in range(200):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        batch_xs = np.reshape(batch_xs,[-1,28,28,1])
        summary,_=sess.run([merged,optimizer], feed_dict={x: batch_xs,y: batch_ys})
        print_stats(sess, batch_xs, batch_ys, cost, accuracy)
        if batch_i%1==0:
            train_writer.add_summary(summary,batch_i)
            summary,_ = sess.run([merged,accuracy],feed_dict={x: x_test_data,y: y_test_data}) 
            test_writer.add_summary(summary,batch_i)

    #saver = tf.train.Saver()
    #save_path = saver.save(sess, save_model_path)



