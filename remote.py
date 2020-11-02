import json
import sys
from ancillary import list_recursive
import numpy as np
import os


def helper(C, D_x):
    C_xx = C[:D_x, :D_x]
    C_yy = C[D_x:, D_x:]
    C_yx = C[D_x:, :D_x]
    C_xy = C_yx.transpose()

    # This computes the two covariance matrices, in which we will take the eigenvalues of.
    C_x1 = np.matmul(np.linalg.pinv(C_xx), C_xy)
    C_x2 = np.matmul(np.linalg.pinv(C_yy), C_yx)
    C_u = np.matmul(C_x1, C_x2)

    C_y1 = np.matmul(np.linalg.pinv(C_yy), C_yx)
    C_y2 = np.matmul(np.linalg.pinv(C_xx), C_xy)
    C_v = np.matmul(C_y1, C_y2)

    # Computing final values of U and V.
    tmp, Uh = np.linalg.eig(C_u)
    tmp, Vh = np.linalg.eig(C_v)
    return Uh, Vh
    
def remote_1(args):
    epsilon_all = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]
    input_list = args["input"]
    
    # Declaring variables...
    
    S = 3
    D_x = 15
    maxItr = 10
    for itr in range(maxItr):
        for eps_id in range(len(epsilon_all)):
            Ch = 0
            Co = 0
            # aggregate the results from sites
            for site in input_list:
                Ch += np.array(input_list[site]["output_val_Ch"])[:, :, eps_id, itr]
                Co += np.array(input_list[site]["output_val_Cs"])
                
            Ch = 1/S * Ch
            Co = 1/S * Co
                
            # Extracting C_xx, C_yy, C_xy, and C_yx from the concatenated matrix.
            Uh, Vh = helper(Ch, D_x)
            Us, Vs = helper(Co, D_x)
            
            myFile = '/output/remote/simulatorRun/result_' + str(itr) + '_' + str(eps_id)
            np.savez(myFile, Uh, Vh, Us, Vs)
#    computation_output = {"output": {"output_listU": Uh.tolist(), "output_listV": Vh.tolist()}, "success": True}
    computation_output = {"output": {}, "success": True}
    
    return json.dumps(computation_output)


if __name__ == '__main__':

    parsed_args = json.loads(sys.stdin.read())
    phase_key = list(list_recursive(parsed_args, 'computation_phase'))

    if "local_1" in phase_key:
        computation_output = remote_1(parsed_args)
        sys.stdout.write(computation_output)
    else:
        raise ValueError("Error occurred at Remote")
