import tensorflow as tf
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
print(r"/////////////////////////////////////////////////")
print(tf.test.is_gpu_available())

