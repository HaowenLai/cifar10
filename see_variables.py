#!/usr/bin/python3
from tensorflow.python import pywrap_tensorflow  
import os

model_dir = '/home/savage/Public/03181059/temp'

checkpoint_path = os.path.join(model_dir, "model.ckpt-1999")  
reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)  
var_to_shape_map = reader.get_variable_to_shape_map()  
for key in var_to_shape_map:  
    print("tensor_name: ", key)  
    #print(reader.get_tensor(key)) # Remove this if you want to print only variable names
