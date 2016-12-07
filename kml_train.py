# -*- coding: utf-8 -*-
from __future__ import print_function

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import RMSprop
import argparse
import datetime
import numpy as np
import matplotlib.pyplot as plt
import kml_utils

SLASH = 0.2 # percentage of test(validation) data

# parsing arguments
def parse_args():
    parser = argparse.ArgumentParser(description='image classifier')
    parser.add_argument('--data', dest='data_dir', default='data')
    parser.add_argument('--list', dest='list_dir', default='list')
    args = parser.parse_args()
    return args

args = parse_args()
if kml_utils.exist_list(args.list_dir):
    print('Lists already exist in ./{0}. Use these lists.'.format(args.list_dir))
    classes, train_list, test_list = kml_utils.load_lists(args.list_dir)
else:
    print('Lists do not exist. Create list from ./{0}.'.format(args.data_dir))
    classes, train_list, test_list = kml_utils.create_list(args.data_dir, args.list_dir, SLASH)

train_image, train_label = kml_utils.load_images(classes, train_list)
test_image, test_label = kml_utils.load_images(classes, test_list)

# convert to numpy.array
x_train = np.asarray(train_image)
y_train = np.asarray(train_label)
x_test = np.asarray(test_image)
y_test = np.asarray(test_label)

print('train samples: ', len(x_train))
print('test samples: ', len(x_test))

NUM_CLASSES = len(classes)
BATCH_SIZE = 32
EPOCH = 100

# building the model
print('building the model ...')

model = Sequential()

model.add(Convolution2D(32, 3, 3, border_mode='valid',
                        input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(64, 3, 3, border_mode='valid'))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(NUM_CLASSES))
model.add(Activation('softmax'))

rmsplop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(loss='categorical_crossentropy', optimizer=rmsplop, metrics=['accuracy'])

# training
hist = model.fit(x_train, y_train,
                 batch_size=BATCH_SIZE,
                 verbose=1,
                 nb_epoch=EPOCH,
                 validation_data=(x_test, y_test))    

# save model
date_str = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
model.save('kml_' + date_str + '.model')

# plot loss
print(hist.history.keys())
loss = hist.history['loss']
val_loss = hist.history['val_loss']
acc = hist.history['acc']
val_acc = hist.history['val_acc']

nb_epoch = len(loss)
fig, ax1 = plt.subplots()
ax1.plot(range(nb_epoch), loss, label='loss', color='b')
ax1.plot(range(nb_epoch), val_loss, label='val_loss', color='g')
leg = plt.legend(loc='upper left', fontsize=10)
leg.get_frame().set_alpha(0.5)
ax2 = ax1.twinx()
ax2.plot(range(nb_epoch), acc, label='acc', color='r')
ax2.plot(range(nb_epoch), val_acc, label='val_acc', color='m')
leg = plt.legend(loc='upper right', fontsize=10)
leg.get_frame().set_alpha(0.5)
plt.grid()
plt.xlabel('epoch')
plt.savefig('graph_' + date_str + '.png')
plt.show()
