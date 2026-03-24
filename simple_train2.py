import time
import os
import numpy as np
from pathlib import Path
from sklearn.neural_network import MLPClassifier
from read_mnist import load_mnist_input, load_mnist_labels
import pickle
from tqdm import tqdm

# TRAINING SCRIPT


def nr_ran0(num):
    IA = 16807
    IM = 2147483647
    AM = 1.0 / np.float32(IM)
    IQ = 127773
    IR = 2836
    MASK = 123459876
    
    num = num ^ MASK
    k = num // IQ
    num = IA * (num - k * IQ) - IR * k
    num = num + IM if num < 0 else num
    rnd0_1 = AM * num
    num = num ^ MASK
    return num, np.float32(rnd0_1)


class SimpleLogNNet2_1:
    digit_len = 28 * 28
    def __init__(self, coefs, intercepts):
        self.coefs = coefs
        self.intercepts = intercepts
        self.rows = coefs.shape[0]
        
    def calc_xwt(self, X):
        ret_value = []
        congnum = 1
        for _ in range(0, self.rows):
            val = np.float32(0.0)
            for j in range(0, self.digit_len):
                congnum, rnd01 = nr_ran0(congnum)
                rnd01 = rnd01 - 0.5
                val = val + X[j] * rnd01
            ret_value.append(val)
        return ret_value
        
    def predict(self, X):
        xwt = self.calc_xwt(X)
        idx_of_max, max_val = -1, 0.0
        for i in range(self.coefs.shape[1]):
            out_i = 0.0
            for j in range(self.rows):
                out_i += self.coefs[j, i] * xwt[j]
            out_i += self.intercepts[i]
            if idx_of_max == -1 or out_i > max_val:
                idx_of_max, max_val = i, out_i
        return idx_of_max


def main():
    input_path = Path('/home/uri-zackhem/mnist/train-images.idx3-ubyte')
    X_train = load_mnist_input(input_path)
    X_train = X_train.reshape(X_train.shape[0], -1).astype(np.float32) / 255.0
    labels_path = Path('/home/uri-zackhem/mnist/train-labels.idx1-ubyte')
    y_train = load_mnist_labels(labels_path)
    test_input_path = Path('/home/uri-zackhem/mnist/t10k-images.idx3-ubyte')
    X_test = load_mnist_input(test_input_path)
    X_test = X_test.reshape(X_test.shape[0], -1).astype(np.float32) / 255.0
    test_labels_path = Path('/home/uri-zackhem/mnist/t10k-labels.idx1-ubyte')
    y_test = load_mnist_labels(test_labels_path)

    features_len = X_train.shape[1]
    n_features = 60
    # output_size = 10
    W = initialize_W_ino2(num_rows_W=n_features, input_dim=features_len)
    
    X_train_wt = np.dot(X_train, W.T)
        
    mlp_params = dict()
    mlp_params['solver'] = 'adam'
    mlp_params['learning_rate'] = 'adaptive'
    mlp_params['early_stopping'] = True
    mlp_params['n_iter_no_change'] = 10
    mlp_params['random_state'] = int.from_bytes(os.urandom(1))
    mlp_params['verbose'] = True
    mlp_params['hidden_layer_sizes'] = ()
    mlp_params['max_iter'] = 5000
    mlp_params['verbose'] = True
    mlp_params['tol'] = 1e-8
    
    cls = MLPClassifier(**mlp_params)
    cls.fit(X_train_wt, y_train)
    X_test_wt = np.dot(X_test, W.T)
    y_pred = cls.predict(X_test_wt)
    accuracy = np.mean(y_pred == y_test)
    print(f'Accuracy: {accuracy:.4f}')
    
    simplelognnet_21 = SimpleLogNNet2_1(cls.coefs_[0], cls.intercepts_[0])
    
    acc21 = []
    for x_test_i, y_test_i in tqdm(zip(X_test, y_test)):
        pred_i = simplelognnet_21.predict(x_test_i)
        acc21.append(pred_i == y_test_i)
        if 0 == len(acc21) % 100:
            print(f'{len(acc21)} | {np.mean(acc21):.4f}')
        if len(acc21) >= 1000:
            break

    dump_path = '/home/uri-zackhem/PycharmProjects/LogNNet-master/simple_model3_results'
    
    ts = int(time.time())
    time_struct = time.localtime(ts)
    dump_name = f'LAST__MLP_model'
    os.makedirs(dump_path, exist_ok=True)
    np.savez(Path(dump_path) / (dump_name + '.npz'), 
             W=W, 
             mlp_coefs=cls.coefs_[0], 
             mlp_intercepts=cls.intercepts_[0],
             test_accuracy=accuracy)
    with open(Path(dump_path) / (dump_name + '.pkl'), 'wb') as f:
        pickle.dump(cls, f)
    

def initialize_W_ino2(num_rows_W: int, 
                     input_dim: int) -> np.ndarray:
    W = np.zeros((num_rows_W, input_dim), np.float32)
    congnum = 1
    for i in range(0, num_rows_W):
        for j in range(0, input_dim):
            congnum, rnd01 = nr_ran0(congnum)
            W[i, j] = rnd01
    W = W - 0.5
    return W


if __name__ == '__main__':
    main()
    print('Finished')
