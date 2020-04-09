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

def draw_acc_subplot(df, subplot, color):
    y = df["acc"]
    x = df["mean_lat"]
    e = df["std_dev_lat"]
    subplot.errorbar(x, y, xerr=e, fmt='.', mfc=color, mec=color,linestyle="", ecolor=[color], label=args.device, color=color)
    subplot.set_title('{} Params vs Latency Behaviour'.format(args.device))
    subplot.set_ylabel('acc(%)', fontsize=13)
    subplot.set_xlabel('latency(ms)', fontsize=13)

    locs = subplot.get_xticks() 
    x_ticklabels = [str(i) for i in locs ]
    subplot.set_xticklabels(x_ticklabels)


def draw_errorbar_graph(file_name, device, path):
    df = pd.read_csv(file_name)
    print(df)
    print("start visualizing {} device {}".format(args.filename, args.device))
    print("result image can be seen in path ./{}".format(args.path))
    
    plt.figure(figsize=(8,5))
    plt.clf()
    labels = []
    figure, axes = plt.subplots(nrows=1, ncols=1)
    color = device_color_code(device)
    draw_acc_subplot(df, axes, color)

    figure.tight_layout(pad=0.3)

    #lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.,labels=labels)
    save_name = path + '/acc_lat_{}'.format(device)
    if (args.latency_only):
        print("latency only graph")
        save_name = path + '/lat_only_{}'.format(device)
    plt.savefig(save_name + ".pdf", ext='pdf', bbox_inches='tight')
    plt.savefig(save_name + ".png", ext='png', bbox_inches='tight')
    plt.show()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filename', nargs='+', type=str, default='arch_profile_cpu.csv', help='start system with test config')
    parser.add_argument('-d', '--device', type=str, default='gpu-rtx2080', help='device used for profile')
    parser.add_argument('-p', '--path', type=str, default='img', help='path to pdf image results')
    parser.add_argument('--latency_only', action='store_true', default=False)
    args = parser.parse_args()
    list_files = args.filename
    if type(list_files) is list:
        for csv_file in list_files:
            draw_errorbar_graph(csv_file, args.device, args.path) 
    else:
        draw_errorbar_graph(list_files, args.device, args.path) 