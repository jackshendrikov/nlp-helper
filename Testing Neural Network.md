# Тестування

Як вже згадувалось вище, ми можемо завантажити навчену мережу LSTM, використовуючи об'єкт Saver Tensorflow. Але перед створенням цього об'єкта ми повинні спочатку створити Tensorflow graph.

Оголосимо деякі з наших гіперпараметрів:


```python
numDimensions = 300
maxSeqLength = 250
batchSize = 24
lstmUnits = 64
numClasses = 2
iterations = 100000
```

Тепер завантажимо наші структури даних:


```python
import numpy as np
wordsList = np.load('wordsList.npy').tolist()
wordsList = [word.decode('UTF-8') for word in wordsList] #Encode words as UTF-8
wordVectors = np.load('wordVectors.npy')
```

Далі створимо наш graph. Це той самий код, який був у попередньому файлі:


```python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()
tf.reset_default_graph()

labels = tf.placeholder(tf.float32, [batchSize, numClasses])
input_data = tf.placeholder(tf.int32, [batchSize, maxSeqLength])

data = tf.Variable(tf.zeros([batchSize, maxSeqLength, numDimensions]),dtype=tf.float32)
data = tf.nn.embedding_lookup(wordVectors,input_data)

lstmCell = tf.nn.rnn_cell.BasicLSTMCell(lstmUnits)
lstmCell = tf.nn.rnn_cell.DropoutWrapper(cell=lstmCell, output_keep_prob=0.25)
value, _ = tf.nn.dynamic_rnn(lstmCell, data, dtype=tf.float32)

weight = tf.Variable(tf.truncated_normal([lstmUnits, numClasses]))
bias = tf.Variable(tf.constant(0.1, shape=[numClasses]))
value = tf.transpose(value, [1, 0, 2])
last = tf.gather(value, int(value.get_shape()[0]) - 1)
prediction = (tf.matmul(last, weight) + bias)

correctPred = tf.equal(tf.argmax(prediction,1), tf.argmax(labels,1))
accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))
```

    WARNING:tensorflow:From C:\Users\johnb\anaconda3\envs\tensorflow\lib\site-packages\tensorflow\python\ops\resource_variable_ops.py:1659: calling BaseResourceVariable.__init__ (from tensorflow.python.ops.resource_variable_ops) with constraint is deprecated and will be removed in a future version.
    Instructions for updating:
    If using Keras pass *_constraint arguments to layers.
    WARNING:tensorflow:From <ipython-input-3-2d14546dcca9>:13: BasicLSTMCell.__init__ (from tensorflow.python.ops.rnn_cell_impl) is deprecated and will be removed in a future version.
    Instructions for updating:
    This class is equivalent as tf.keras.layers.LSTMCell, and will be replaced by that in Tensorflow 2.0.
    WARNING:tensorflow:From <ipython-input-3-2d14546dcca9>:15: dynamic_rnn (from tensorflow.python.ops.rnn) is deprecated and will be removed in a future version.
    Instructions for updating:
    Please use `keras.layers.RNN(cell)`, which is equivalent to this API
    WARNING:tensorflow:From C:\Users\johnb\anaconda3\envs\tensorflow\lib\site-packages\tensorflow\python\ops\rnn_cell_impl.py:738: Layer.add_variable (from tensorflow.python.keras.engine.base_layer_v1) is deprecated and will be removed in a future version.
    Instructions for updating:
    Please use `layer.add_weight` method instead.
    WARNING:tensorflow:From C:\Users\johnb\anaconda3\envs\tensorflow\lib\site-packages\tensorflow\python\ops\rnn_cell_impl.py:744: calling Zeros.__init__ (from tensorflow.python.ops.init_ops) with dtype is deprecated and will be removed in a future version.
    Instructions for updating:
    Call initializer instance with the dtype argument instead of passing it to the constructor
    

Тепер завантажуємо в мережу:


```python
sess = tf.InteractiveSession()
saver = tf.train.Saver()
saver.restore(sess, tf.train.latest_checkpoint('models'))
```

    INFO:tensorflow:Restoring parameters from models\pretrained_lstm.ckpt-90000
    

Перш ніж ми введемо наш власний текст, спочатку визначимо пару функцій. Перша - це функція, для переконання, що речення у відповідному форматі, а друга - це функція, яка отримує слова-вектори для кожного зі слів у даному реченні:


```python
#Видаляє пунктуацію, круглі дужки, знаки питання і т.д. Лишає тільки буквенно-цифрові символи
import re
strip_special_chars = re.compile("[^A-Za-z0-9 ]+")

def cleanSentences(string):
    string = string.lower().replace("<br />", " ")
    return re.sub(strip_special_chars, "", string.lower())

def getSentenceMatrix(sentence):
    arr = np.zeros([batchSize, maxSeqLength])
    sentenceMatrix = np.zeros([batchSize,maxSeqLength], dtype='int32')
    cleanedSentence = cleanSentences(sentence)
    split = cleanedSentence.split()
    for indexCounter,word in enumerate(split):
        try:
            sentenceMatrix[0,indexCounter] = wordsList.index(word)
        except ValueError:
            sentenceMatrix[0,indexCounter] = 399999 # вектор для невідомих слів
    return sentenceMatrix
```

Тепер ми можемо створити наш вхідний текст:


```python
inputText = "That movie was terrible."
inputMatrix = getSentenceMatrix(inputText)
```


```python
inputText = "That movie was terrible."
inputMatrix = getSentenceMatrix(inputText)

predictedSentiment = sess.run(prediction, {input_data: inputMatrix})[0]
# predictedSentiment[0] представляє вихідний бал для позитивного настрою
# predictedSentiment[1] представляє вихідний бал для негативного настрою

if (predictedSentiment[0] > predictedSentiment[1]):
    print("Позитивний настрій")
else:
    print ("Негативний настрій")
```

    Негативний настрій
    


```python
secondInputText = "Truly a masterpiece, The Best Hollywood film of 2019, one of the Best films of the decade..."
secondInputMatrix = getSentenceMatrix(secondInputText)
```


```python
predictedSentiment = sess.run(prediction, {input_data: secondInputMatrix})[0]
if (predictedSentiment[0] > predictedSentiment[1]):
    print("Позитивний настрій")
else:
    print ("Негативний настрій")
```

    Позитивний настрій
    
