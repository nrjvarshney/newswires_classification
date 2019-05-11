
# coding: utf-8

# In[1]:


from keras.datasets import reuters


# In[2]:


(train_data, train_labels), (test_data, test_labels) = reuters.load_data(
num_words=10000)


# In[3]:


# num_words restricts data to the 10000 most frequently occuring words 


# In[5]:


len(train_data),len(test_data)


# In[6]:


train_data[0]


# In[7]:


# decoding bact to text
word_index = reuters.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
decoded_newswire = ' '.join([reverse_word_index.get(i - 3, '?') for i in
train_data[0]])


# In[8]:


decoded_newswire


# In[10]:


import numpy as np
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)


# In[13]:


x_train[0],len(x_train[0])


# In[32]:


train_labels[0]


# In[14]:


def to_one_hot(labels, dimension=46):
    results = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
        results[i, label] = 1.
    return results

one_hot_train_labels = to_one_hot(train_labels)
one_hot_test_labels = to_one_hot(test_labels)


# In[17]:


one_hot_train_labels,len(one_hot_train_labels),len(one_hot_train_labels[0])


# In[18]:


# there is a built-in way to do this in Keras 
from keras.utils.np_utils import to_categorical

one_hot_train_labels = to_categorical(train_labels)
one_hot_test_labels = to_categorical(test_labels)


# # Building your network

# In[19]:


from keras import models
from keras import layers
model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))




# In[20]:



model.compile(optimizer='rmsprop',
loss='categorical_crossentropy',
metrics=['accuracy'])


# In[21]:




x_val = x_train[:1000]
partial_x_train = x_train[1000:]
y_val = one_hot_train_labels[:1000]
partial_y_train = one_hot_train_labels[1000:]


# In[22]:



history = model.fit(partial_x_train,
partial_y_train,
epochs=20,
batch_size=512,
validation_data=(x_val, y_val))



# In[23]:


import matplotlib.pyplot as plt
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[24]:


plt.clf()
acc = history.history['acc']
val_acc = history.history['val_acc']
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[25]:


# network begins to overfit after nine epochs.
# Let's train a new network from scratch for nine epochs and then evaluate

model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))

model.compile(optimizer='rmsprop',
loss='categorical_crossentropy',
metrics=['accuracy'])

model.fit(partial_x_train,
partial_y_train,
epochs=9,
batch_size=512,
validation_data=(x_val, y_val))

results = model.evaluate(x_test, one_hot_test_labels)


# In[26]:


results


# In[28]:


predictions = model.predict(x_test)


# In[29]:


predictions[0]


# In[30]:


np.sum(predictions[0])


# In[31]:


np.argmax(predictions[0])


# In[33]:


y_train = np.array(train_labels)
y_test = np.array(test_labels)


# In[34]:


y_train[0]


# In[35]:


model.compile(optimizer='rmsprop',
loss='sparse_categorical_crossentropy',
metrics=['acc'])


# In[36]:


model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(4, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))

model.compile(optimizer='rmsprop',
loss='categorical_crossentropy',
metrics=['accuracy'])

model.fit(partial_x_train,
partial_y_train,
epochs=20,
batch_size=128,
validation_data=(x_val, y_val))


# The network now peaks at ~71% validation accuracy, an 8% absolute drop. This drop
# is mostly due to the fact that you’re trying to compress a lot of information (enough
# information to recover the separation hyperplanes of 46 classes) into an intermediate
# space that is too low-dimensional.

# # Further Experiments
# Try using larger or smaller layers: 32 units, 128 units, and so on.
#  You used two hidden layers. Now try using a single hidden layer, or three hid-
# den layers.

# # Notes
# 1. If you’re trying to classify data points among N classes, your network should end with a Dense layer of size N .
# 
# 2. In a single-label, multiclass classification problem, your network should end with a softmax activation so that it will output a probability distribution over the N output classes.
# 
# 3. Categorical crossentropy is almost always the loss function you should use for such problems. It minimizes the distance between the probability distributions output by the network and the true distribution of the targets.
# 
# 4. There are two ways to handle labels in multiclass classification:
#     – Encoding the labels via categorical encoding (also known as one-hot encoding) and using categorical_crossentropy as a loss function
#     – Encoding the labels as integers and using the sparse_categorical_crossentropy loss function
# 
# 5. If you need to classify data into a large number of categories, you should avoid creating information bottlenecks in your network due to intermediate layers that are too small.
