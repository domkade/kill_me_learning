# -*- coding: utf-8 -*-
from __future__ import print_function

from keras.models import load_model
import argparse
import kml_utils
import os
import numpy as np

# parsing arguments
def parse_args():
    parser = argparse.ArgumentParser(description='image classifier')
    parser.add_argument('--data', dest='data_dir', default='data')
    parser.add_argument('--list', dest='list_dir', default='list')
    parser.add_argument('--model', dest='model_name', required=True)
    args = parser.parse_args()
    return args
    
args = parse_args()
if kml_utils.exist_list(args.list_dir):
    print('Lists exist in ./{0}. Use the test list.'.format(args.list_dir))
    classes, _, test_list = kml_utils.load_lists(args.list_dir)
else:
    print('Lists do not exist.')
    exit(0)

test_image, test_label = kml_utils.load_images(classes, test_list)

# convert to numpy.array
x_test = np.asarray(test_image)
y_test = np.asarray(test_label)

print('test samples: ', len(x_test))

model = load_model(args.model_name)

pred = model.predict(x_test, batch_size=32, verbose=0)
for i, test in enumerate(test_list):
    print(test)
    answer = os.path.basename(os.path.dirname(test))
    predict = classes[np.argmax(pred[i])]
    if answer == predict:
        print('Correct!!!')
    else:
        print('Incorrect...')
    print('answer:', os.path.basename(os.path.dirname(test)))
    print('predict:', classes[np.argmax(pred[i])])
    for j in range(len(classes)):
        print('{0}: {1:4.2f} '.format(classes[j], pred[i][j]), end='')
    print('\n')
