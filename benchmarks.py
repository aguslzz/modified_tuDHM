import utilities as ut
from itertools import product
import pandas as pd

cost_functions = {'std': ut.binarized_hologram_std_cf, 
                'maxsum': ut.binarized_hologram_maxsum_cf
            }

optimization_methods = ["Nelder-Mead", "L-BFGS-B", "TNC", "SLSQP", "Powell", "trust-constr", "COBYLA"]

holograms = {
    'glioblastoma': ('Glioblastoma.jpg', 0.533, 3.75),
    'holo-RBC': ('holo-RBC-20p205-2-3.png', 0.532, 2.4),
    'holo': ('holo.jpg', 0.633, 3.75),
    'uofm': ('hologram UofM.png', 0.532, 2.4)
}

reconstruction_function = {
    'CFS': ut.reconstruction,
    'VortexCFS': ut.reconstruction_vortex_v2
}

for params in product(holograms.items(), cost_functions.items(), optimization_methods, reconstruction_function.items()):
    print(params[0])
    print(params[1])
    print(params[2])
    print(params[3])
    print(params[3][1])
    print('---------------------------------------')
    
    reconstruction_func = params[3][1]
    
    
    
