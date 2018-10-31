import os
import tensorflow as tf

#Turn off logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#Initialize tf variables 
a = tf.get_variable('a', shape=[1])
b = tf.get_variable('b', shape=[1])

#perform a tf operation
c = tf.add(a,b)

#Start the session
with tf.Session() as session:
	
	#Initialize all variables
    session.run(tf.global_variables_initializer())

    #Initialize writer for logging
    writer = tf.summary.FileWriter(".", session.graph)
    
    #run the session for value c
    result = session.run(c, feed_dict={a:[4],b:[6]})

    print(result)
    writer.close()