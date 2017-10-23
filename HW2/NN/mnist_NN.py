
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
valid_data = mnist.validation.next_batch(5000)
test_data = mnist.test.next_batch(10000)

x_valid, y_valid = valid_data
x_valid = np.reshape(x_valid,[-1,784])
x_test, y_test = test_data
x_test = np.reshape(x_test,[-1,784])


def nn_model(features):
    with tf.name_scope('input_layer'):
        input_layer = features
    with tf.name_scope('fc1_layer'):
        fc1 = tf.layers.dense(input_layer,units=200)
    with tf.name_scope('fc2_layer'):
        fc2 = tf.layers.dense(fc1,units=50)
    with tf.name_scope('fc3_layer'):
        fc3 = tf.layers.dense(fc2,units=10)
    return fc3

#save_model_path = './'
with tf.name_scope('intputs'):
    x = tf.placeholder(tf.float32,[None,784],name='x')
with tf.name_scope('targets'):
    y = tf.placeholder(tf.float32,[None,10],name='y')
with tf.name_scope('logits'):
    logits = nn_model(x)
with tf.name_scope('cost'):
    loss = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=logits)
    cost = tf.reduce_mean(loss)
with tf.name_scope('train'):
    optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost)


with tf.name_scope('accuracy'):
    correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1),name='correct_pred')
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32),name='accuracy')
tf.summary.scalar('accuracy', accuracy)

merged = tf.summary.merge_all()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    train_writer = tf.summary.FileWriter('./train',sess.graph) 
    test_writer = tf.summary.FileWriter('./test') 
    for i in range(20000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        batch_xs = np.reshape(batch_xs,[-1,784])
        summary,_ = sess.run([merged,optimizer],feed_dict={x: batch_xs,y: batch_ys})    
        if i%100 ==0:
            train_writer.add_summary(summary,i)
            summary,_ = sess.run([merged,accuracy],feed_dict={x: x_test,y: y_test}) 
            test_writer.add_summary(summary,i)

    test_writer.close()
    train_writer.close()



