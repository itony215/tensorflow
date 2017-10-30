from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import tensorflow as tf
x = tf.placeholder(tf.float32, [None, 784]) #placeholder占用位子 None代表可以有很多張圖片 784每張圖有784維向量
W = tf.Variable(tf.zeros([784, 10])) # 權重  值得注意的是 W 的形狀為 [784, 10] 因為我們想要把一個 784 維的向量經由矩陣相乘後產生一個 10 維的證據 (evidence）向量來表示不同的數字
b = tf.Variable(tf.zeros([10]))      # 偏移值 b 則是一個長度為 10 的向量，然後我們可以把他加入最後的輸出中
y = tf.nn.softmax(tf.matmul(x, W) + b) #模型只要一行定義
# 驗證loss
y_ = tf.placeholder(tf.float32, [None, 10]) #為了實現 cross-entropy 我們必須先加入一個新的佔位子 (placeholder) 來放置正確的答案．
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
cross_entropy = tf.reduce_mean(
  tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
  
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy) # 使用梯度下降法 gradient descent algorithm 來最小化 cross_entropy
sess = tf.InteractiveSession() #利用 Session 來初始化我們的參數以及啟動我們的模型了
tf.global_variables_initializer().run()

#執行 1000 次的訓練 每一次 loop 中我們會從訓練數據中隨機抓取一批 100 筆數據
for _ in range(1000): 
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

#評估我們的模型
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1)) #tf.argmax(y, 1) 代表著模型對於每一筆輸入認為最有可能的數字，tf.argmax(y_, 1) 則是代表著正確的數字．我們可以使用 tf.equal 來確認我們的預測是否正確．
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) #轉成布林值
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

#出來的結果大概是 92%

#定義好給下面用
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)
#convolution pooling卷積 池化 stride步伐  
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

#filter特徵篩選器 		 		 		
W_conv1 = weight_variable([5, 5, 1, 32]) #一二三維的數字 [5, 5, 1] 代表著這個過濾器是一個 5x5x1 的矩陣 32代表著我們建立了 32 個過濾器來篩選特徵
b_conv1 = bias_variable([32]) #第四行建立一個偏移值 (bias) 避免負數

#用意是把圖片像素輸入變成一個 1x28x28x1 的四維矩陣
x_image = tf.reshape(x, [-1, 28, 28, 1]) #-1為自動調整 輸入的維度是多少就是多少 28*28像素 1是指黑白色階channel 彩色的話是3 RGB

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)
#結果是 14 x 14 x 32 的輸出

#第二卷積層 5 x 5 的過濾器但是會產生 64 個輸出
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

#全連結層
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

#為了減少 overfitting，在輸出層之前我們會加入 dropout 層
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#最後我們加上像之前 softmax regression 一樣的層． 輸出層
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

#訓練以及評估模型
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)  #把 gradient descent 最佳化演算法換成更為精密的 ADAM 最佳化演算法
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  for i in range(20000): # 20,000 回合的訓練
    batch = mnist.train.next_batch(50)
    if i % 100 == 0:
      train_accuracy = accuracy.eval(feed_dict={
          x: batch[0], y_: batch[1], keep_prob: 1.0})
      print('step %d, training accuracy %g' % (i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
  print('test accuracy %g' % accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
#在 feed_dict 參數之中加入 keep_prob 這個參數來控制 dropout 的機率
#在每一百個回合的訓練中印出紀錄
#這裡就是兩個卷積層再接上一個全連結層．而卷積層又可以分成 convolution 以及 max_pooling