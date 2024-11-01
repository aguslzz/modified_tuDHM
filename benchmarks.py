import utilities as ut
from itertools import product
import pandas as pd
from PIL import Image
import numpy as np
import subprocess
import platform

save_excel = True

operating_system = platform.system()



cost_functions = {'std': ut.binarized_hologram_std_cf, 
                'maxsum': ut.binarized_hologram_maxsum_cf
            }

optimization_methods = ["Nelder-Mead", "L-BFGS-B", "TNC", "SLSQP", "Powell", "trust-constr", "COBYLA"]


holograms = {
    'glioblastoma': ('Glioblastoma.jpg', 0.533, 3.75, 0),
    'holo-RBC': ('holo-RBC-20p205-2-3.png', 0.532, 2.4, 0),
    'holo': ('holo.jpg', 0.633, 3.75, 0),
    'uofm': ('hologram UofM.png', 0.532, 2.4, 0)
}

reconstruction_function = {
    'CFS': ut.reconstruction,
    'VortexCFS': ut.reconstruction_vortex_v2
}

def get_processor_model_linux():
    with open('/proc/cpuinfo', 'r') as f:
        for line in f:
            if 'model name' in line:
                return line.split(':')[1].strip()
            
def get_processor_model_windows():
    try:
        output = subprocess.check_output("wmic cpu get name", shell=True)
        # Decode and clean the output
        model = output.decode().split('\n')[1].strip()
        return model
    except Exception as e:
        return str(e)
    
if operating_system == 'Linux':
    processor = get_processor_model_linux()
elif operating_system == 'Windows':
    processor = get_processor_model_windows()
else:
    processor = 'Unknown'


df_columns = ['Hologram Name',
                  'Hologram Path',
                  'Wavelength',
                  'Pixel Size',
                  'Distance',
                  'Cost Function',
                  'Reconstruction Method',
                  'Optimization Method',
                  'Opti. Time',
                  'Fx Sol',
                  'Fy Sol',
                  'J Sol',
                  'AmplitudeSolPath',
                  'PhaseSolPath',
                  'CPU'
                ]
    
benchmark_data = pd.DataFrame(columns=df_columns)

for params in product(holograms.items(), cost_functions.items(), optimization_methods, reconstruction_function.items()):    
    reconstruction_func = params[3][1]
    reconstruction_func_name = params[3][0]
    cost_function = params[1][1]
    cost_function_name = params[1][0]
    hologram_path = params[0][1][0]
    hologram = ut.imageRead(params[0][1][0])
    holo_name = params[0][0]
    wavelength = params[0][1][1]
    pixel_size = params[0][1][2]
    distance = params[0][1][3]
    opti_method = params[2]
    
    complexObject, res, time_elapsed = reconstruction_func(
                                            field=hologram, 
                                            wavelength=wavelength, 
                                            dxy=pixel_size, 
                                            distance=distance,
                                            method=opti_method, 
                                            cf=cost_function)
    
    amplitude = ut.amplitude(complexObject, False)
    phase = ut.phase(complexObject)
    
    fx_sol = res.x[0]
    fy_sol = res.x[1]
    
    J = res.fun
    
    save_phase_path = f'./results/{holo_name}/{holo_name}_{cost_function_name}_{opti_method}_phase.png'
    save_amplitude_path = f'./results/{holo_name}/{holo_name}_{cost_function_name}_{opti_method}_amplitude.png'
    
    # Save the results as images
    # ut.imageSave does not exist
    # this does not work since amplitude is numpy array
    
    amplitude_image = Image.fromarray(amplitude.astype(np.int32))
    phase_image = Image.fromarray(phase.astype(np.int32))
    
    amplitude_image.save(save_amplitude_path, mode='I')
    phase_image.save(save_phase_path, mode='I')
    
    new_row = pd.Series({
        'Hologram Name': holo_name,
        'Hologram Path': hologram_path,
        'Wavelength': wavelength,
        'Pixel Size': pixel_size,
        'Distance': distance,
        'Cost Function': cost_function_name,
        'Reconstruction Method': reconstruction_func_name,
        'Optimization Method': opti_method,
        'Opti. Time': time_elapsed,
        'Fx Sol': fx_sol,
        'Fy Sol': fy_sol,
        'J Sol': J,
        'AmplitudeSolPath': save_amplitude_path,
        'PhaseSolPath': save_phase_path,
        'CPU': processor,  # Add CPU information
    })

    # Assuming benchmark_data DataFrame already exists
    benchmark_data = benchmark_data.append(new_row, ignore_index=True)
    
    
if save_excel:
    benchmark_data.to_excel('./results/benchmark_results.xlsx', index=False)
   
    
    
