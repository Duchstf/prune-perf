import hls4ml
import os
import yaml

import tensorflow as tf
def load_model(file_path):
    from qkeras.utils import _add_supported_quantized_objects
    co = {}
    _add_supported_quantized_objects(co)  
    return tf.keras.models.load_model(file_path, custom_objects=co)

def yaml_load(config):
        with open(config, 'r') as stream:
            param = yaml.safe_load(stream)
        return param

#Plot settings
import matplotlib.pyplot as plt
import mplhep as hep
plt.style.use(hep.style.ROOT)

import matplotlib.pylab as pylab
params = {'legend.fontsize': 'medium',
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'medium',
         'ytick.labelsize':'medium'}
pylab.rcParams.update(params)

#line thickness
import matplotlib as mpl
mpl.rcParams['lines.linewidth'] = 5

#For looping over files
import glob

#----------------------------------------------

model_dirs = [f.path for f in os.scandir('../anomaly_models/') if f.is_dir()]

for model_dir in model_dirs:
    
    print('Converting: ', model_dir)
    pruned_percent = model_dir.split("_")[-1][:-7]
    
    model = load_model(model_dir + '/model_ToyCar.h5')
    
    os.environ['PATH'] += os.pathsep + '/data/Xilinx/Vivado/2022.2/bin'

    #
    hls4ml.model.optimizer.get_optimizer('output_rounding_saturation_mode').configure(layers=['Activation'], rounding_mode='AP_RND', saturation_mode='AP_SAT')
    
    config = hls4ml.utils.config_from_keras_model(model, granularity='name')
    config['Model']['Strategy'] = 'Latency'
    config['Model']['Precision'] = 'ap_fixed<16,8>'
    config['LayerName']['input_1']['Precision'] = 'ap_fixed<8,8>'

    for Layer in config['LayerName'].keys():    
        if "Dense" in Layer:
            config['LayerName'][Layer]['ReuseFactor'] = 1

        else:
            config['LayerName'][Layer]['ReuseFactor'] = 1
            

    hls_model = hls4ml.converters.convert_from_keras_model(model,
                                                           backend='Vitis',
                                                           project_name='anomaly', #I'm not very creative
                                                           clock_period=5, #1/360MHz = 2.8ns
                                                           hls_config=config,
                                                           output_dir='hardware/{}_pruned/hls4ml_prj'.format(pruned_percent),
                                                           part='xcvu9p-flga2104-2L-e')

    hls_model.compile()

    hls_model.build(csim=False, reset = True)
    