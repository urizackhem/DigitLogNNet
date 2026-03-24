from pathlib import Path
import numpy as np

# CONVERTING numpy arrays to C++ h-file constants.

def make_line(arr, factor=None):
    str_format = "{fval:.6f}f" if factor is None else "{int(np.round(fval * factor))}"
    str_format = "f'" + str_format + "'"
    line = '{'
    for i in range(arr.shape[0]):
        fval = arr[i]
        line += eval(str_format)
        if i != arr.shape[0] - 1:   
            line += ', '
    line += '}'
    return line



def dump_h1(path,
            name,
            arr,
            arrname,
            factor=None):
    with open(Path(path) / (name + '.h'), 'w', newline='') as f:
        f.write('#pragma once\n\n')
        f.write(f'#define {arrname}_LEN {arr.shape[0]}\n\n')            
        if factor is not None:
            f.write(f'#define {arrname}_FACTOR {int(factor)}\n\n')
            f.write(f'const int   {arrname}[{arrname}_LEN] = ')
        else:
            f.write(f'const float {arrname}[{arrname}_LEN] = ')
        line = make_line(arr, factor)
        f.write(line)
        f.write(';\n\n')


def dump_h2(path,
            name,
            arr,
            arrname,
            factor=None):
    with open(Path(path) / (name + '.h'), 'w', newline='') as f:
        f.write('#pragma once\n\n')
        f.write(f'#define {arrname}_ROWS {arr.shape[0]}\n\n')
        f.write(f'#define {arrname}_COLS {arr.shape[1]}\n\n')
        if factor is not None:
            f.write(f'#define {arrname}_FACTOR {int(factor)}\n\n')
            f.write(f'const int   {arrname}[{arrname}_ROWS][{arrname}_COLS] = {{\n')
        else:
            f.write(f'const float {arrname}[{arrname}_ROWS][{arrname}_COLS] = {{\n')
        for i in range(arr.shape[0]):
            line = make_line(arr[i], factor)
            if i != arr.shape[0] - 1:
                line += ','
            line += '\n'        
            f.write(line)
        f.write('};\n\n')


if __name__ == '__main__':
    in_dir = ('/home/uri-zackhem/PycharmProjects/'
              'LogNNet-master/simple_model2_results')
    file_name = 'LAST__MLP_model.npz'
    npz_path = Path(in_dir) / file_name
    data = np.load(npz_path)
    print(data)
    mlp_coefs = data['mlp_coefs']
    mlp_intercepts = data['mlp_intercepts']
    dump_h2(in_dir, 'mlp_coefs', mlp_coefs, 'MLP_COEFS', factor=10_000)
    dump_h1(in_dir, 'mlp_intercepts', mlp_intercepts, 'MLP_INTERCEPTS', factor=10_000)
    print('Finished')
