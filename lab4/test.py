import tensorflow as tf

# Check if GPU is available and visible to TensorFlow
if tf.config.list_physical_devices('GPU'):
    print("GPU is available")
else:
    print("GPU is NOT available")

# Simple GPU computation
with tf.device('/GPU:0'):  # Use GPU:0 if available
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0])
    b = tf.constant([5.0, 4.0, 3.0, 2.0, 1.0])
    c = a * b

# Print the result
print("Result of GPU computation:")
print(c)
