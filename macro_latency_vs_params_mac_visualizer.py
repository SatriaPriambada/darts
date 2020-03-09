import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import pandas as pd
import numpy as np

from matplotlib.patches import Patch

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--filename', nargs='+', type=str, default='gen_latencies_architecture_cpu-i7-4578U.csv', help='start system with test config')
parser.add_argument('-d', '--device', type=str, default='cpu-i7-4578U', help='device used for profile')
parser.add_argument('-p', '--path', type=str, default='img', help='path to pdf image results')
args = parser.parse_args()

def device_color_code():
    if args.device == 'cpu-i7-4578U':
        return "red"
    elif args.device == 'gpu-rtx2080':
        return "green"
    elif args.device == 'gpu-v100':
        return "blue"
    elif args.device == 'cpu-x6132':
        return "brown"

def draw_param_subplot(df, subplot, color):
    x = df["params"]
    y = df["mean_lat"]
    e = df["std_dev_lat"]
    subplot.errorbar(x, y, yerr=e, fmt='.', mfc=color, mec=color,linestyle="", ecolor=[color], label=args.device, color=color)
    subplot.set_title('{} Params vs Latency Behaviour'.format(args.device))
    subplot.set_xlabel('params(M)', fontsize=13)
    subplot.set_ylabel('latency(ms)', fontsize=13)

    locs = subplot.get_xticks() 
    print(locs)
    x_ticklabels = [str(i/1000000) for i in locs ]
    subplot.set_xticklabels(x_ticklabels)

def draw_mac_subplot(df, subplot, color):
    x = df["macs"]
    y = df["mean_lat"]
    e = df["std_dev_lat"]
    subplot.errorbar(x, y, yerr=e, fmt='.', mfc=color, mec=color,linestyle="", ecolor=[color], label=args.device, color=color)
    subplot.set_title('{} MACS vs Latency Behaviour'.format(args.device))
    subplot.set_xlabel('macs(M)', fontsize=13)
    subplot.set_ylabel('latency(ms)', fontsize=13)

def draw_errorbar_graph(df):
    print(df)
    print("start visualizing {} device {}".format(args.filename, args.device))
    print("result image can be seen in path ./{}".format(args.path))
    
    plt.figure(figsize=(8,5))
    plt.clf()
    labels = []
    figure, axes = plt.subplots(nrows=2, ncols=1)
    color = device_color_code()
    for i,row in enumerate(axes):
        if i == 0:
            draw_param_subplot(df, row, color)
        elif i == 1:
            draw_mac_subplot(df, row, color)

    figure.tight_layout(pad=0.3)

    #lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.,labels=labels)
    plt.savefig(args.path + '/params_lat_{}.pdf'.format(args.device), ext='pdf', bbox_inches='tight')
    plt.savefig(args.path + '/params_lat_{}.png'.format(args.device), ext='png', bbox_inches='tight')
    plt.show()
    

if __name__ == '__main__':
    list_files = args.filename
    for csv_file in list_files:
        csv_path = csv_file
        df = pd.read_csv(csv_path)
        draw_errorbar_graph(df) 