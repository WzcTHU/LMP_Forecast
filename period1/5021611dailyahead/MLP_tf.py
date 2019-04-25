# -*-coding:utf-8-*-
import numpy
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import random
from func import *
import matplotlib.pyplot as plt

scaler1 = StandardScaler()
scaler2 = StandardScaler()
sess = tf.InteractiveSession()
# 0702_0902LMP_2
# [x_train, y_train, x_train_noload, load_train] = get_train_data('0702_0902LMP_2.csv', '0702_0902load_2.csv')
# [x_test, y_test, x_test_noload, load_test] = get_train_data('test_lmp.csv', 'test_load.csv')
x_train, y_train, x_test, y_test, spike_train = divide_data('0702_1001_lmp.csv', '0702_1001_load.csv')
scaler1.fit(x_train)
scaler2.fit(y_train)
x_train = scaler1.transform(x_train)
y_train = scaler2.transform(y_train)
x_test = scaler1.transform(x_test)
y_test = scaler2.transform(y_test)

in_units = 9
h1_units = 120
h2_units = 120
out_units = 1

W1 = tf.Variable(tf.truncated_normal([in_units, h1_units], stddev=0.1), name='W1')
b1 = tf.Variable(tf.truncated_normal([h1_units], stddev=0.1), name='b1')
W2 = tf.Variable(tf.truncated_normal([h1_units, h2_units], stddev=0.1), name='W2')
b2 = tf.Variable(tf.truncated_normal([h2_units], stddev=0.1), name='b2')
W3 = tf.Variable(tf.truncated_normal([h2_units, out_units], stddev=0.1), name='W3')
b3 = tf.Variable(tf.truncated_normal([out_units], stddev=0.1), name='b3')
saver = tf.train.Saver()

x_in = tf.placeholder(tf.float32, [None, in_units], name='x_in')
# keep_prob = tf.placeholder(tf.float32)
hi_layer1 = tf.nn.relu(tf.matmul(x_in, W1) + b1)
hi_layer2 = tf.nn.relu(tf.matmul(hi_layer1, W2) + b2)

y_cal = tf.matmul(hi_layer2, W3) + b3
y_ = tf.placeholder(tf.float32, [None, out_units], name='y_')

with tf.name_scope('loss'):
    rmse = tf.sqrt(tf.reduce_sum((y_cal - y_) ** 2))
    tf.summary.scalar('loss', rmse)

writer = tf.summary.FileWriter('log_MLP/', sess.graph)

merged = tf.summary.merge_all()

learning_rate = tf.Variable(0.01, dtype=tf.float32)
train_step = tf.train.AdamOptimizer(0.01).minimize(rmse)

init = tf.global_variables_initializer()
print('Initializing...')
sess.run(init)
for count in range(100001):
    [batch_xs, batch_ys] = make_batch(x_train, y_train, batch_size=200)
    _, summary = sess.run([train_step, merged], feed_dict={x_in: batch_xs, y_: batch_ys})
    if((count % 50) == 0):
        print(count)
        # sess.run(tf.assign(learning_rate, learning_rate * 1))
        writer.add_summary(summary, count)
save_path = saver.save(sess, 'para/1.ckpt')
#--------------------------------------------------------------------------
y_fore_train = sess.run(y_cal, feed_dict={x_in: x_train})
y_fore_test = sess.run(y_cal, feed_dict={x_in: x_test})

y_fore_train = scaler2.inverse_transform(y_fore_train)
y_train = scaler2.inverse_transform(y_train)

y_fore_test = scaler2.inverse_transform(y_fore_test)
y_test = scaler2.inverse_transform(y_test)

print('the error on train data is:', cal_mape(y_train, y_fore_train))
print('the error on test data is:', cal_mape(y_test, y_fore_test))

fig1 = plt.figure()
plt.title('2018.9.22---2018.10.1 PJM NODE:5021072 DAY AHEAD LMP FORECASTING')
l1 = plt.plot(y_test, marker='*', label='actual')
l2 = plt.plot(y_fore_test, marker='o', label='forecast')
plt.legend()
plt.show()
