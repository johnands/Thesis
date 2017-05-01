import tensorflow as tf

# NN parameters
inputs = 3
outputs = 1
nodes = 10

# declare input and output tensors
x = tf.placeholder(tf.float32, [None,inputs])
y = tf.placeholder(tf.float32, [None,outputs])

# weights and biases input layer -> hidden layer
W1 = tf.Variable( tf.random_normal([inputs,nodes]) )
b1 = tf.Variable( tf.constant(0.1, shape=[nodes]) )
preActivation = tf.add(tf.matmul(x, W1), b1) 
activation1 = tf.nn.sigmoid(preActivation)

# weights and biases hidden layer -> output layer
W2 = tf.Variable( tf.random_normal([nodes,outputs]) )
b2 = tf.Variable( tf.constant(0.1, shape=[outputs]) )
preActivation = tf.add(tf.matmul(activation1, W2), b2) 
output = preActivation

# start session
with tf.Session() as sess:
    
    # initialize variables
    sess.run( tf.global_variables_initializer() )
    
    # write summaries
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('Summaries', sess.graph)
    
    print sess.run(output, feed_dict={x: [[1.0, 2.0, 3.0]]})
    


