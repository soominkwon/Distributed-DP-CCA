import json
import os
import sys
from ancillary import list_recursive
from scipy.io import loadmat
import numpy as np

def local_1(args):

    input_list = args["input"]
    myFile = input_list["covariates"]
    
    tmp = loadmat(os.path.join(args["state"]["baseDirectory"], myFile))
    X_s = tmp['Xs']
    Y_s = tmp['Ys']
    Z_s = np.concatenate([X_s, Y_s])
    D, N_s = Z_s.shape

    C_s = 1/N_s * (np.matmul(Z_s, Z_s.transpose()))
    
    # Constants for Gaussian noise:
    epsilon_all = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]
    delta = 0.01
    maxItr = 10
    C_hat = np.zeros(shape = [D, D, len(epsilon_all), maxItr])

    for itr in range(maxItr):
        for eps_id in range(len(epsilon_all)):
            epsilon = epsilon_all[eps_id]
            sigma = (1/(N_s * epsilon)) * np.sqrt(2 * np.log(1.25/delta))
            temp = np.random.normal(0, sigma, size = [D, D])
            temp2 = np.triu(temp);
            temp3 = temp2.T;
            temp4 = np.tril(temp3, -1);
            E = temp2 + temp4;
        
            # Addition of noise to C matrix:
            C_hat[:, :, eps_id, itr] = C_s + E

    computation_output = {
        "output": {
            "output_val_Ch": C_hat.tolist(),
            "output_val_Cs": C_s.tolist(),
            "computation_phase": 'local_1'
        }
    }

    return json.dumps(computation_output)


if __name__ == '__main__':

    parsed_args = json.loads(sys.stdin.read())
    phase_key = list(list_recursive(parsed_args, 'computation_phase'))

    if not phase_key:
        computation_output = local_1(parsed_args)
        sys.stdout.write(computation_output)
    else:
        raise ValueError("Error occurred at Local")
