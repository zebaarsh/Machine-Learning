
# coding: utf-8

# In[16]:


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

#************Load data*******************************
mnist = tf.contrib.learn.datasets.load_dataset("mnist")
train_data = mnist.train.images     # Returns np.array
train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
test_data = mnist.test.images       # Returns np.array
test_labels = np.asarray(mnist.test.labels, dtype=np.int32)



# In[39]:


#******Model function for implementing CNN*********
def cnn_funct(features,labels,mode):
    input_layer = tf.reshape(features["x"],[-1,28,28,1])

#******Convolutional Layer 1*********
    conv_1 = tf.layers.conv2d(inputs = input_layer, filters=32, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)

#******Pooling Layer 1****************
    pool_1 = tf.layers.max_pooling2d(inputs = conv_1, pool_size=[2, 2], strides=2)

#******Convolutional Layer 2*********
    conv_2 = tf.layers.conv2d(inputs = pool_1, filters=64, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)

#******Pooling Layer 2****************
    pool_2 = tf.layers.max_pooling2d(inputs = conv_2, pool_size=[2, 2], strides=2)

#******Dense Layer********************
    pool2_flat = tf.reshape(pool_2, [-1, 7 * 7 * 64])
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

#******Logits Layer*******************
    logits = tf.layers.dense(inputs=dropout, units=10)

#******Generate predictions (for PREDICT and TEST mode) and add `softmax_tensor` to the graph. It is used for PREDICT and by the `logging_hook`.
    predictions = {"classes": tf.argmax(input=logits, axis=1),
                   "probabilities": tf.nn.softmax(logits, name="softmax_tensor")}
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

#******Calculate Loss (for both TRAIN and TEST modes)***********
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)

#******Configure the Training Op (for TRAIN mode)***************
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(loss=loss,global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

#*******Add evaluation metrics (for TEST mode)******************
    eval_metric_ops = {"accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)    


# In[40]:


#*************Create the model estimator which performs classification***********************************
#def model_classifier(model_funct):
mnist_classifier = tf.estimator.Estimator(model_fn=cnn_funct, model_dir="./Project 3- Classification/mnist_convnet_model")


# In[41]:


#********Set up logging to log the probability values from the softmax layer of our CNN*************
tensors_to_log = {"probabilities": "softmax_tensor"}
logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)


# In[34]:


#********Train the model****************************
def train_model(x):
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": train_data},
    y=train_labels,
    batch_size=100,
    num_epochs=None,
    shuffle=True)
    mnist_classifier.train(input_fn=train_input_fn,steps=1000,hooks=[logging_hook])
    


# In[ ]:


#*********Test the model and print results***********
def test_model(x):
    test_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": test_data},
    y=test_labels,
    num_epochs=1,
    shuffle=False)
    test_results = mnist_classifier.evaluate(input_fn=test_input_fn)
    print(test_results)
    


# In[35]:


train_model(train_data)


# In[42]:


test_model(test_data)

