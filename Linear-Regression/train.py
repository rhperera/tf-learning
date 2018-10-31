import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#Initialize placeholders for input and output data
Y = tf.placeholder(tf.float32, shape=[None, 1])
X = tf.placeholder(tf.float32, shape=[None, 1])

#Initialize theta and c, our parameters in the hypothesis as variables
theta = tf.get_variable('theta', shape=[1], initializer=tf.ones_initializer())
c = tf.get_variable('c', shape=[1], initializer=tf.ones_initializer())

#Define our hypothesis
h = tf.add(tf.multiply(X, theta), c)

#Define our cost
cost = tf.reduce_mean(tf.squared_difference(Y, h))

#Let's optimize our cost using Gradient Descent
optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(cost)

#The dictionary which our training data is held
feed_dict = {X:[[1],[2],[3],[4]],Y:[[3],[5],[7],[9]]}

#Define a scalar for cost for logging
cost_summary = tf.summary.scalar('Cost', cost)
merged = tf.summary.merge_all()

#number of training rounds
n = 1000

with tf.Session() as session:
    #initialize all variables
    session.run(tf.global_variables_initializer())

    #Initialize writer for logging and log the Graph
    writer = tf.summary.FileWriter('.', session.graph)

    #Let's start our training for n rounds
    for i in range(n):
        '''
        Run the session to calculate optimizer value from Graph
        We feed in our input and output data dictionary here
        We also run merged to get our scalar values for logging 
        '''
        _, cst_summary = session.run([optimizer, merged], feed_dict= feed_dict)

        #Let's write these logs with the global step number
        writer.add_summary(cst_summary, global_step=i)

    #print the values of trained theta and c
    print("Value of theta is " + str(theta.eval()))
    print("Value of c is " + str(c.eval()))

    #Let's now run a prediction and print it
    pred = session.run(h, feed_dict={X:[[10]]})
    print("Prediction if x=10, y="+ str(pred[0]))
    
    writer.flush()
    writer.close()
