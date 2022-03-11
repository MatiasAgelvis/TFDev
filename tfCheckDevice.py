# Check if the server/ instance is having GPU/ CPU from python code
import sys
import tensorflow as tf
from tensorflow.python.client import device_lib

## this command list all the processing device GPU and CPU
device_lib.list_local_devices()

device_name = [x.name for x in device_lib.list_local_devices() if x.device_type == 'GPU']
if device_name and device_name[0] == "/device:GPU:0":
    device_name = "/gpu:0"
    print('GPU')
else:
    print('CPU')
    device_name = "/cpu:0"
