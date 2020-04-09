import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import pandas as pd
import numpy as np

from matplotlib.patches import Patch

def device_color_code(device):
    if device == 'cpu-i7-4578U':
        return "red"
    elif device == 'gpu-rtx2080':
        return "green"
    elif device == 'gpu-v100':
        return "blue"
    elif device == 'cpu-x6132':
        return "brown"

def draw_hist_subplot(df, subplot, color, n_bins, latency_types):
    for lat_type in latency_types:
        x = df[lat_type]
        subplot.hist(x, bins=n_bins, alpha=0.5, label=lat_type)    
        subplot.set_title('{} Latency Behaviour'.format(args.device))
        subplot.set_ylabel('count', fontsize=13)
        subplot.set_xlabel('latency(ms)', fontsize=13)

        locs = subplot.get_xticks() 
        x_ticklabels = [str(i) for i in locs ]
        subplot.set_xticklabels(x_ticklabels)


def draw_histogram_graph(file_name, device, path):
    df = pd.read_csv(file_name)
    print(df)
    print("start visualizing {} device {}".format(args.filename, args.device))
    print("result image can be seen in path ./{}".format(args.path))
    
    plt.figure(figsize=(8,5))
    plt.clf()
    labels = ["mean_lat", "lat95", "lat99"]
    n_bins = 100
    latency_types = ["mean_lat", "lat95", "lat99"]
    figure, axes = plt.subplots(nrows=1, ncols=1)
    color = device_color_code(device)
    if axes is list:
        for i, subplot in enumerate(axes):
            draw_hist_subplot(df, subplot, color, n_bins, latency_types)
    else:
        draw_hist_subplot(df, axes, color, n_bins, latency_types)

    figure.tight_layout(pad=0.3)
    lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.,labels=labels)

    save_name = path + '/lat_only_{}'.format(device)
    plt.savefig(save_name + ".pdf", ext='pdf',bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.savefig(save_name + ".png", ext='png',bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.show()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filename', nargs='+', type=str, default='arch_profile_imagenet_gpu-rtx2080.csv', help='start system with test config')
    parser.add_argument('-d', '--device', type=str, default='gpu-rtx2080', help='device used for profile')
    parser.add_argument('-p', '--path', type=str, default='img', help='path to pdf image results')
    args = parser.parse_args()
    list_files = args.filename
    if type(list_files) is list:
        for csv_file in list_files:
            draw_histogram_graph(csv_file, args.device, args.path) 
    else:
        draw_histogram_graph(list_files, args.device, args.path) 