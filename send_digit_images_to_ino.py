import os
from pathlib import Path
import numpy as np
import serial
import time
from random import shuffle, seed


def load_mnist_input(path):
    with open(path, 'rb') as f:
        intType = np.dtype('int32').newbyteorder('>')
        nMetaDataBytes = 4 * intType.itemsize
        data = np.fromfile(path, dtype='ubyte')
        magicBytes, nImages, width, height = np.frombuffer(data[:nMetaDataBytes].tobytes(), intType)
        data = data[nMetaDataBytes:].reshape([nImages, width, height])
    return data


def load_mnist_labels(path):
    with open(path, 'rb') as f:
        intType = np.dtype('int32').newbyteorder('>')
        nMetaDataBytes = 2 * intType.itemsize
        data = np.fromfile(path, dtype='ubyte')[nMetaDataBytes:]
    return data


def main():
    input_path = Path('/home/uri-zackhem/mnist/t10k-images.idx3-ubyte')
    data = load_mnist_input(input_path)
    labels_path = Path('/home/uri-zackhem/mnist/t10k-labels.idx1-ubyte')
    labels = load_mnist_labels(labels_path)
    
    arduino = serial.Serial(port='/dev/ttyACM0', 
                            baudrate=9600)
    num_images = 100
    seed(int.from_bytes(os.urandom(1)))
    shuffled_indices = list(range(len(data)))
    shuffle(shuffled_indices)
    acc_array = []
    for idx, i in enumerate(shuffled_indices):
        if idx >= num_images:
            break
        digit_image = data[i].flatten()
        digit_label = labels[i]
        arduino.reset_input_buffer()
        arduino.write(bytes(digit_image.tolist()))
        time.sleep(0.1)
        msg = arduino.readline()
        print(msg)
        retval = int(arduino.read())
        acc_array.append(retval == digit_label)
        print(f'Label: {digit_label}, Predicted: {retval}')

    print(f'Accuracy: {np.mean(acc_array):.4f}')


if __name__ == '__main__':
    main()
    print('Finished')
