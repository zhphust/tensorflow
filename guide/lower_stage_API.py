# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

# 定义常量Tensor
a = tf.constant(3.0, dtype=tf.float32)
b = tf.constant(4.0)              # also tf.float32 implicitly
total = a + b
print(a)                          # tf.Tensors不具有值，它们只是计算图中元素的手柄。
print(b)
print(total)

# 保存计算图为TensorBoard摘要文件，需要时从Terminal读取
writer = tf.summary.FileWriter('.')
writer.add_graph(tf.get_default_graph())

# 创建会话
sess = tf.Session()
print(sess.run(total))
print(sess.run({'ab': (a, b), 'total': total}))

vec = tf.random_uniform(shape=(3,))
out1 = vec + 1
out2 = vec + 2
print(sess.run(vec))
print(sess.run(vec))
print(sess.run((out1, out2)))

# 占位符
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
z = x + y
print(sess.run(z, feed_dict={x: 3, y: 4.5}))
print(sess.run(z, feed_dict={x: [1, 3], y: [2, 4]}))


# 数据集
my_data = [
    [0, 1, ],
    [2, 3, ],
    [4, 5, ],
    [6, 7, ],
]
slices = tf.data.Dataset.from_tensor_slices(my_data)
next_item = slices.make_one_shot_iterator().get_next()

while True:
    try:
        print(sess.run(next_item))
    except tf.errors.OutOfRangeError:
        break

r = tf.random_normal([3, 3])
dataset = tf.data.Dataset.from_tensor_slices(r)
iterator = dataset.make_initializable_iterator()
next_row = iterator.get_next()

sess = tf.Session()
sess.run(iterator.initializer)
while True:
    try:
        print(sess.run(next_row))
    except tf.errors.OutOfRangeError:
        break

# 层
x = tf.placeholder(tf.float32, shape=[None, 3])
linear_model = tf.layers.Dense(units=1)
y = linear_model(x)

# 初始化层
init = tf.global_variables_initializer()
sess.run(init)

# 执行层
print(sess.run(y, {x: [[1, 2, 3],[4, 5, 6]]}))

# 层函数的快捷方式
x = tf.placeholder(tf.float32, shape=[None, 3])
y = tf.layers.dense(x, units=1)

init = tf.global_variables_initializer()
sess.run(init)

print(sess.run(y, {x: [[1, 2, 3], [4, 5, 6]]}))


# 特征列
features = {
    'sales': [[5], [10], [8], [9]],
    'department': ['sports', 'sports', 'gardening', 'gardening']}

department_column = tf.feature_column.categorical_column_with_vocabulary_list(
    'department', ['sports', 'gardening'])
department_column = tf.feature_column.indicator_column(department_column)

columns = [
    tf.feature_column.numeric_column('sales'),
    department_column
]

inputs = tf.feature_column.input_layer(features, columns)

# 初始化特征列
var_init = tf.global_variables_initializer()
table_init = tf.tables_initializer()
sess = tf.Session()
sess.run((var_init, table_init))

print(sess.run(inputs))

# 完整程序
# 定义数据
x = tf.constant([[1], [2], [3]], dtype=tf.float32)
y_true = tf.constant([[0], [-1], [-2]], dtype=tf.float32)

# 定义模型
layer = tf.layers.Dense(units=1)
y_pred = layer(x)

# 定义损失函数
loss = tf.losses.mean_squared_error(labels=y_true, predictions=y_pred)

# 定义优化器
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

# 初始化
init = tf.global_variables_initializer()

# 定义Session
sess = tf.Session()
sess.run(init)

# 开始训练
for i in range(100):
    _, loss_value = sess.run((train, loss))
    print(loss_value)

# 预测
print(sess.run(y_pred))

