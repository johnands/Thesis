import tensorflow as tf

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
    
outputData = tf.placeholder(tf.float32, [None,outputs], name='outputData')

# cost function
with tf.name_scope('cost'):
    error = tf.subtract(actOutput, outputData, name='deviation')
    trainCost = tf.nn.l2_loss( error, name='L2norm')

# optimizer
with tf.name_scope('optimizer'):
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001, name='Adam')
    trainStep = optimizer.minimize(trainCost, name='trainStep')
    
initOperation = tf.global_variables_initializer()

# start session
with tf.Session() as sess:
    
    # initialize all variables
    sess.run(initOperation)
    
    # write summaries
    #tf.summary.FileWriter('Summaries', sess.graph)
    
    # get data
    #xBatch, yBatch = getNextBatch()
    
    print sess.run( trainCost, feed_dict={inputData: [[1.0, 2.0, 3.0]], outputData: [[3.0]]} )
    
    for i in xrange(1000):

        # train
        sess.run( trainStep, feed_dict={inputData: [[1.0, 2.0, 3.0]], outputData: [[3.0]]} )
        
        if not i % 100:
            print sess.run( trainCost, feed_dict={inputData: [[1.0, 2.0, 3.0]], outputData: [[3.0]]} )
        


