import tensorflow as tf

# NN parameters
inputs = 3
nodes = 10
outputs = 1

with tf.name_scope('inputLayer'):
    inputData = tf.placeholder(tf.float32, [None,inputs], name='inputData')

with tf.name_scope('layer1'):
    
    W1 = tf.Variable( tf.random_normal([inputs,nodes]), name='weights' )
    b1 = tf.Variable( tf.constant(0.1, shape=[nodes]), name='biases' )
 
    preAct1 = tf.add( tf.matmul(inputData, W1), b1, name='preActivation' ) 
    act1 = tf.nn.sigmoid(preAct1, name='activation')

with tf.name_scope('outputLayer'):
    
    W2 = tf.Variable( tf.random_normal([nodes,outputs]), name='weights' )
    b2 = tf.Variable( tf.constant(0.1, shape=[outputs]), name='biases' )

    preActOutput = tf.add( tf.matmul(act1, W2), b2, name='preActivation' ) 
    actOutput = tf.identity(preActOutput, name='activation')
    
# initialization node
initOperation = tf.global_variables_initializer()

# start session
with tf.Session() as sess:
    
    # initialize all variables
    sess.run(initOperation)
    
    # write summaries
    tf.summary.FileWriter('Summaries', sess.graph)
    


