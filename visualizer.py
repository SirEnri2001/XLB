import numpy as np
import matplotlib.pyplot as plt
from termcolor import colored
import jaxlib.xla_extension

def visualize_data(data, field_name='u', ts_start=0, ts_stride=1, ts_end=-1, img_num=20):
    if ts_end == -1:
        ts_end = len(data['timestep'])
    ts_range = data['timestep'][-1] - data['timestep'][0] + 1
    uniformed_ts_start = (ts_start - data['timestep'][0])/ts_range
    uniformed_ts_end = (ts_end - data['timestep'][0])/ts_range
    uniformed_ts_stride = max(ts_stride/ts_range, (uniformed_ts_end - uniformed_ts_start)/img_num)
    idx_num=len(data['timestep'])
    idx_list=range(int(uniformed_ts_start*idx_num),int(uniformed_ts_end*idx_num),int(uniformed_ts_stride*idx_num))
    fig, axs = plt.subplots(1, len(idx_list), figsize=(32, 10))
    max_val=data[field_name].max()
    min_val=data[field_name].min()
    for i, ax in zip(idx_list, axs if len(idx_list)>1 else [axs]):
        img = data[field_name][i]
        img = (img - min_val)/(max_val - min_val)
        img = np.concatenate((img, np.zeros((img.shape[0], img.shape[1], 1))), axis=2)
        ax.imshow(img)
        ax.set_title("T={}".format(data['timestep'][i]))

def inspect_data(data):
    print(colored("*** Data Summary ***", "red"))
    if len(data['timestep'])==1:
        print(colored("Single frame at timestep: {}".format(data['timestep'][0]), "green"))
        return
    else:
        print(colored("Timestep: first frame at T={} - last frame at T={}".format(data['timestep'][0], data['timestep'][-1]), 'blue'))
        print(colored("Timestep stride: {}".format(data['timestep'][1]-data['timestep'][0]), 'blue'))
    print(colored("Frame attributes:", 'blue'))
    for i in data.keys():
        print(colored('\t{}'.format(i), 'green'), end=" ")
        if len(data[i])==0:
            print(colored("No data", 'red'))
        elif data[i][0] is None:
            print(None)
        elif type(data[i][0])==np.ndarray:
            print("numpy array shape: {}".format(data[i][0].shape))
        elif type(data[i][0])==jaxlib.xla_extension.ArrayImpl:
            print("jax array shape: {}".format(data[i][0].shape))
        elif type(data[i][0])==int:
            print("int")
        else:
            print(type(data[i][0]))