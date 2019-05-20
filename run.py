import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler


def load_data():
    data = pd.read_csv("data.csv", delimiter=',', dtype='a')
    return data.values

def regression(data):
    sess = tf.Session()
    x1_vals   = data[:,0].astype(np.float32)
    x2_vals   = data[:,1].astype(np.float32)
    y_vals    = data[:,2].astype(np.float32)

    x1_data   = tf.placeholder(shape=[None,1], dtype=tf.float32)
    x2_data   = tf.placeholder(shape=[None,1], dtype=tf.float32)
    y_target  = tf.placeholder(shape=[None,1], dtype=tf.float32)

    X1        = tf.Variable(tf.random_normal(shape=[1,1]))
    X2        = tf.Variable(tf.random_normal(shape=[1,1]))

    model_output = tf.add(tf.matmul(x1_data, X1), tf.matmul(x2_data, X2))

    loss = tf.reduce_mean(tf.abs(y_target - model_output))

    init = tf.global_variables_initializer()
    sess.run(init)
    my_opt = tf.train.GradientDescentOptimizer(0.001)
    train_step = my_opt.minimize(loss)

    for i in range(5000):
        rand_index = np.random.choice(len(x1_vals), size=100)
        rand_x1 = np.transpose([x1_vals[rand_index]])
        rand_x2 = np.transpose([x2_vals[rand_index]])
        rand_y = np.transpose([y_vals[rand_index]])

        sess.run(train_step, feed_dict={x1_data: rand_x1, x2_data: rand_x2, y_target: rand_y})
        temp_loss = sess.run(loss, feed_dict={x1_data: rand_x1, x2_data: rand_x2, y_target: rand_y})
        if i%25 == 0:
            print(f'Step {i} - Loss {temp_loss} - X1 = {sess.run(X1)}, X2 = {sess.run(X2)}')

    return (sess.run(X1), sess.run(X2))

def main():
    data = load_data()
    result = regression(data)
    print(f'X1={result[0][0]}, X2={result[1][0]}')

if __name__ == "__main__":
    main()