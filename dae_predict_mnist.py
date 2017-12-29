import numpy as np
import matplotlib.pyplot as plt

from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.layers import (Input, Dense, Concatenate)
from keras.datasets import mnist
from keras.callbacks import TensorBoard
from keras.utils import np_utils

log_dir = 'C:/Users/dawidkot/Documents/Python Scripts/Python Medium/DAE/tmp/tb'
log_dir2 = 'C:/Users/dawidkot/Documents/Python Scripts/Python Medium/DAE/tmp/tb2'
# cd C:/Users/dawidkot/Documents/Python Scripts/Python Medium/DAE
# tensorboard --logdir="C:/Users/dawidkot/Documents/Python Scripts/Python Medium/DAE/tmp/tb"
# http://localhost:6006/

batch_size = 128
epochs = 20
noise_factor = 0.5
model_fname = 'dae_deep'

# input image dimensions
img_rows, img_cols = 28, 28                          
input_shape = (img_rows * img_cols, )

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], np.prod(x_train.shape[1:]))
x_test = x_test.reshape(x_test.shape[0], np.prod(x_test.shape[1:]))

x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)
x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)

x_feat_train = np.concatenate((x_train, x_test), axis=0)
x_feat_train_noisy = np.concatenate((x_train_noisy, x_test_noisy), axis=0)

print(x_feat_train_noisy.shape[0], ' dae train samples')
    
def dae_deep(features_shape, act='relu'):

    # Input
    x = Input(name='inputs', shape=features_shape, dtype='float32')
    o = x
    
    # Encoder / Decoder
    o = Dense(1024, activation=act, name='dense1')(o)
    o = Dense(1024, activation=act, name='dense2')(o)
    o = Dense(1024, activation=act, name='dense3')(o)
    dec = Dense(784, activation='sigmoid', name='dense_dec')(o)
    
    # Print network summary
    Model(inputs=x, outputs=dec).summary()
    
    return Model(inputs=x, outputs=dec)

callbacks = [TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=False),
             ModelCheckpoint(monitor='loss',
                             filepath=model_fname + '.hdf5',
                             save_best_only=True,
                             save_weights_only=True,
                             mode='min')]
             
autoenc = dae_deep(input_shape)
autoenc.compile(optimizer='adadelta', loss='binary_crossentropy')

autoenc.fit(x_feat_train_noisy, x_feat_train,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
            callbacks=callbacks)

decoded_imgs = autoenc.predict(x_feat_train_noisy)

n = 10
plt.figure(figsize=(20, 4))
for i in range(1,n+1):
    # display original
    ax = plt.subplot(2, n, i)
    plt.imshow(x_feat_train_noisy[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

def dae_features(model, model_fname):
    model.load_weights(model_fname)
    input_ = model.get_layer('inputs').input
    feat1 = model.get_layer('dense1').output
    feat2 = model.get_layer('dense2').output
    feat3 = model.get_layer('dense3').output
    feat = Concatenate(name='concat')([feat1, feat2, feat3])
    model = Model(inputs=[input_],
                      outputs=[feat])
    return model

_model = dae_features(autoenc, model_fname + '.hdf5')
features_train = _model.predict(x_train)
features_test = _model.predict(x_test)
print(features_train.shape, ' train samples shape')
print(features_test.shape, ' train samples shape')

def deep_nn(features_shape, num_classes, act='relu'):

    # Input
    x = Input(name='inputs', shape=features_shape, dtype='float32')
    o = x
    
    # Encoder / Decoder
    o = Dense(1024, activation=act, name='dense1')(o)
    o = Dense(1024, activation=act, name='dense2')(o)
    o = Dense(1024, activation=act, name='dense3')(o)
    y_pred = Dense(num_classes, activation='sigmoid', name='pred')(o)
    
    # Print network summary
    Model(inputs=x, outputs=y_pred).summary()
    
    return Model(inputs=x, outputs=y_pred)

input_shape2 = (features_train.shape[1], )
num_classes = 10

y_train_ohe = np_utils.to_categorical(y_train, num_classes)
y_test_ohe = np_utils.to_categorical(y_test, num_classes)

callbacks = [TensorBoard(log_dir=log_dir2, histogram_freq=0, write_graph=False)]
             
deep = deep_nn(input_shape2, num_classes)
deep.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['acc'])

deep.fit(features_train, y_train_ohe,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
            validation_data=(features_test, y_test_ohe),
            callbacks=callbacks)

# The predict_classes function outputs the highest probability class
# according to the trained classifier for each input example.
predictions = deep.predict(features_test)
predicted_classes = np.argmax(predictions, axis=1)

# Check which items we got right / wrong
correct_indices = np.nonzero(predicted_classes == y_test)[0]
incorrect_indices = np.nonzero(predicted_classes != y_test)[0]

plt.figure()
for i, correct in enumerate(correct_indices[:9]):
    plt.subplot(3,3,i+1)
    plt.imshow(x_test[correct].reshape(28,28), cmap='gray', interpolation='none')
    plt.title("Predicted {}, Class {}".format(predicted_classes[correct], y_test[correct]))
    
plt.figure()
for i, incorrect in enumerate(incorrect_indices[:9]):
    plt.subplot(3,3,i+1)
    plt.imshow(x_test[incorrect].reshape(28,28), cmap='gray', interpolation='none')
    plt.title("Predicted {}, Class {}".format(predicted_classes[incorrect], y_test[incorrect]))