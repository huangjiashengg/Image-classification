# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

# 本实验采用fasion_mnist数据集，该数据集包含 10 个类别
# 的 70,000 个灰度图像。这些图像以低分辨率（28x28 像素）
# 展示了单件衣物

# 加载数据集
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
# print(train_images.shape)
# print(len(train_labels))
# print(train_labels)
# print(test_images.shape)
# print(len(test_labels))
# 将输入数据标准化至0-1范围内
train_images = train_images / 255.0
test_images = test_images / 255.0
# Plt构建画布规格
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    # 设置坐标轴刻度
    plt.xticks([])
    plt.yticks([])
    # 设置网格 grid参数
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    # 横坐标加上标注
    plt.xlabel(class_names[train_labels[i]])
# 构建神经网络需要先配置模型的层，然后再编译模型
# 配置模型层
# 该网络的第一层 tf.keras.layers.Flatten 将图像格式从二维数组（28 x 28 像素）
# 转换成一维数组（28 x 28 = 784 像素）。将该层视为图像中未堆叠的像素行并将其排
# 列起来。该层没有要学习的参数，它只会重新格式化数据。
# 展平像素后，网络会包括两个 tf.keras.layers.Dense 层的序列。它们是密集连接或全
# 连接神经层。第一个 Dense 层有 128 个节点（或神经元）。第二个（也是最后一个）层
# 会返回一个长度为 10 的 logits 数组。每个节点都包含一个得分，用来表示当前图像属于 10 个类中的哪一类。
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10)
])
# 编译模型，损失函数，优化器（决定模型如何根据其看到的数据和自身的损失函数进行更新。）
# 指标-用于监控训练和测试步骤
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
# 训练模型
model.fit(train_images, train_labels, epochs=10)
# 模型在测试集上的预测误差
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print('\nTest accuracy:', test_acc)
# 附加一个 softmax 层，将 logits 转换成更容易理解的概率
probability_model = tf.keras.Sequential([model,
                                         tf.keras.layers.Softmax()])
# 预测值，每一个预测值对象都是一个数组，表示该被预测对象属于各个类别的概率
predictions = probability_model.predict(test_images)
print(np.argmax(predictions[0]) == test_labels[0])

###################################################################################
# 将每一个被预测对象的预测结果绘制成图表
def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array, true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array, true_label[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions[i], test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()
