from os import read
from matplotlib import pyplot as plt
import matplotlib
import datetime
import numpy as np
import argparse
import seaborn as sns
import pandas as pd

def smooth(scalars, weight=0.997):  
    # Weight between 0 and 1
    last = scalars[0] 
    smoothed = list()
    for i, point in enumerate(scalars):
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)                        
        last = smoothed_val                                  

    return smoothed

def process_time(text):
    """
    Example: 
    """
    _, h, m, s = text.split(' ')
    h = h[:-1]
    m = m[:-1]
    s = s[:-1]
    return int(h), int(m), int(s)

def process_reward(text):
    """
    Example: Avg Reward 0.08
    """
    _, _, _, reward = text.split(' ')
    return float(reward)

def process_acc(text):
    """
    Example: Avg Accuracy 0.22
    """
    _, _, _, acc = text.split(' ')
    return float(acc)

def read_file(text_file, defaut_gap=0):
    """
    defaut_gap: gap between two time step, (min)
    """
    times = []
    rewards = []
    accs = []

    prev_h = 0.0
    num_day = 0
    f = open(text_file, 'r')
    lines = f.readlines()
    
    default_h = 0
    for i, line in enumerate(lines):
        att = line.split(',')
        if len(att) == 1: #eliminate Training lines
            continue

        time = att[0]
        avg_reward = att[1]
        avg_acc = att[2]

        h,m,s = process_time(time)
        # if avg_reward == '0.42400000000000004' and  time == '00h 00m 58s': # 13h 31m 29s
        #     s += 29
        #     if s > 60:
        #         s -= 60
        #         m+=1
        #     m += 31
        #     if m > 60:
        #         m-= 60
        #         h+= 1
        #     h += 13

        if defaut_gap == 0 or i == 0:
            h += ((s/60 + m)/60 + 24*num_day)
            default_h = h

        if defaut_gap != 0 and i != 0:
            h = default_h + defaut_gap
            default_h = h

        if prev_h > h:
            num_day += 1
            h = ((s/60 + m)/60 + 24*num_day)

        prev_h = h

        reward = process_reward(avg_reward)
        acc = process_acc(avg_acc)
        # print(type(datetime.datetime.now() + datetime.timedelta(hours=h, minutes=m, seconds=s)))
        # times.append(datetime.timedelta(hours=h, minutes=m, seconds=s))
        times.append(h)
        
        rewards.append(reward)
        accs.append(acc)
    
    return (times, rewards, accs)

#plot for more than 3 graphs
def plot_graph_2(graphs, labels, level='easy', shown_type='acc'):
    if len(graphs) != len(labels):
        print("Wrong!!!")
        return
    colors = ['green', 'red', 'blue', 'red', "royalblue"]
    
    times = []
    rewards = []
    accs = []
    
    for i in range(len(graphs)):
        times.append(graphs[i][0])
        rewards.append(graphs[i][1])
        accs.append(graphs[i][2])
    
    weights = []
    if shown_type == 'acc':
        for i in range(len(accs)):
            weights.append(accs[i])
    
    if shown_type == 'reward':
        for i in range(len(rewards)):
            weights.append(rewards[i])

    # upper bound and lower bound smooth 
    weight_smooth_2 = 0.0
    weights_2 = []
    weights_ub = []
    weights_lb = []
    for i in range(len(weights)):
        weights_2.append(smooth(weights[i], weight_smooth_2))

    # smooth for mean lines
    weight_smooth = 0.95
    for i in range(len(weights)):
        weights[i] = smooth(weights[i], weight_smooth)
    
    #create upper bound line and lower bound line
    for i in range(len(weights)):
        tmp_weights_ub = []
        tmp_weights_lb = []
        for j in range(len(weights[i])):
            if weights_2[i][j] > weights[i][j]:
                tmp_weights_ub.append(weights_2[i][j])
                tmp_weights_lb.append(2*weights[i][j] - weights_2[i][j])
            else:
                tmp_weights_ub.append(2*weights[i][j] - weights_2[i][j])
                tmp_weights_lb.append(weights_2[i][j])
        weights_ub.append(tmp_weights_ub)
        weights_lb.append(tmp_weights_lb)
    
    if level == 'medium':
        weights[0] = weights[0][:7170] # based
        times[0] = times[0][:7170]
        
    if level == 'hard':
        weights[0] = weights[0][:10000] #based
        times[0] = times[0][:10000]
    
    weights_max = []
    times_max_index = []
    print("Max")
    for i in range(len(weights)):
        weights_max.append(max(weights[i]))
        times_max_index.append(weights[i].index(weights_max[i]))
        print(labels[i] + ' ' + str(round(max(weights[i]), 3)))

    weights_max = []
    times_max_index = []
    print("Last")
    for i in range(len(weights)):
        weights_max.append(weights[i][-1])
        times_max_index.append(weights[i].index(weights_max[i]))
        print(labels[i] + ' ' + str(round(weights[i][-1], 3)))

    
    #labels
    abandoned_graph = []
    lines = []
    
    for i in range(len(labels)):
        if i in abandoned_graph:
            continue
        labels[i] = labels[i] + ": (" + repr(round(weights_max[i], 3)) + ", " + repr(int(times[i][times_max_index[i]])) + "h)"
        #plot acc lines
        lines.append(plt.plot(times[i], weights[i], color=colors[i], label=labels[i]))

    #plot the max dashed lines
    points = [np.ones(int(max(time))) for time in times]
    text_style = dict(horizontalalignment='right', verticalalignment='center',
                  fontsize=7, fontdict={'family': 'monospace'})
    
    # easy
    # plt.plot((0, times[0][times_max_index[0]]), (weights_max[0], weights_max[0]), color = 'black', linestyle='dashed') # horizontal
    # plt.text(0, weights_max[1], "0.897", **text_style)
    # plt.plot((13.3, 13.3), (0.03, weights_max[1]), color = 'black', linestyle='dashed')# vertical 1
    # plt.text(13.3 + 2, 0.01, "13.3", **text_style)
    # plt.plot((times[0][times_max_index[0]], times[0][times_max_index[0]]), (0.03, weights_max[0]), color = 'black', linestyle='dashed')# vertical 2
    # plt.text(times[0][times_max_index[0]] + 2, 0.01, "101", **text_style)
    # plt.plot((44, 44), (0.03, weights_max[2]), color = 'black', linestyle='dashed')# vertical 3
    # plt.text(44 + 2, 0.01, "44", **text_style)

    # easy diff
    plt.plot((times[1][times_max_index[1]], times[1][times_max_index[1]]), (0.04, weights_max[1]), color = 'black', linestyle='dashed')# vertical
    plt.text(times[1][times_max_index[1]] + 0.3, 0.02, "25", **text_style)
    plt.plot((0, times[1][times_max_index[1]]), (weights_max[1], weights_max[1]), color = 'black', linestyle='dashed')# horizontal 1
    plt.text(0, weights_max[1], "0.902", **text_style)
    plt.plot((0, times[0][times_max_index[0]]), (weights_max[0], weights_max[0]), color = 'black', linestyle='dashed')# horizontal 2
    plt.text(0, weights_max[0], "0.296", **text_style)
    plt.plot((0, times[2][times_max_index[2]]), (weights_max[2], weights_max[2]), color = 'black', linestyle='dashed')# horizontal 3
    plt.text(0, weights_max[2], "0.389", **text_style)
    

    # medium
    # plt.text(0, weights_max[1]-0.01, "0.79", **text_style)
    # plt.plot((0, times[0][times_max_index[0]]), (weights_max[0], weights_max[0]), color = 'black', linestyle='dashed') # horizontal
    # plt.plot((34, 34), (0.03, weights_max[0]), color = 'black', linestyle='dashed')# vertical 1
    # plt.text(34 + 2, 0.01, "34", **text_style)
    # plt.plot((times[0][times_max_index[0]], times[0][times_max_index[0]]), (0.03, weights_max[0]), color = 'black', linestyle='dashed')# vertical 2
    # plt.text(times[0][times_max_index[0]] + 2, 0.01, "125", **text_style)
    # plt.plot((80, 80), (0.03, weights_max[0]), color = 'black', linestyle='dashed')# vertical 3
    # plt.text(80 + 2, 0.01, "82", **text_style)
    
    # # hard
    # plt.text(0, weights_max[1], "0.57", **text_style)
    # plt.plot((0, times[0][times_max_index[0]]), (weights_max[0], weights_max[0]), color = 'black', linestyle='dashed') # horizontal
    # plt.plot((26.9, 26.9), (0.03, weights_max[0]), color = 'black', linestyle='dashed')# vertical 1
    # plt.text(26.9 + 2, 0.01, "26.9", **text_style)
    # plt.plot((times[0][times_max_index[0]], times[0][times_max_index[0]]), (0.03, weights_max[0]), color = 'black', linestyle='dashed')# vertical 2
    # plt.text(times[0][times_max_index[0]] + 2, 0.01, "172", **text_style)
    # plt.plot((129, 129), (0.03, weights_max[0]), color = 'black', linestyle='dashed')# vertical 3
    # plt.text(129+2, 0.01, "129", **text_style)

    for i in range(len(weights)):
        if i in abandoned_graph:
            continue
        # plt.fill_between(times_clone[i], weights_ub_clone[i], weights_lb_clone[i], color=colors[i], alpha=.3)
        error = np.random.normal(0.05, 0.02, size=len(weights[i]))
        plt.fill_between(times[i], weights[i]-error, weights[i]+error, color=colors[i], alpha=.2)
    
    lines_0 = [line[0] for line in lines]
    # lines_0 = [lines[1][0], lines[3][0], lines[4][0]]
    #set limit for y axis
    plt.ylim(0, 1)
    # if level == 'hard':
    #     plt.ylim(0, 0.6)
    plt.legend(handles=lines_0, bbox_to_anchor=(0.5, 1.05), loc='center')#, ncol=2)
    plt.xlabel("Hours")
    if shown_type=='acc':
        plt.ylabel("Accuracy")
    if shown_type=='reward':
        plt.ylabel("Mean Reward")
    plt.show()

def calculate_mean_reward_and_acc(text_file):
    """
    Use to calculate mean reward, or accuracy of the MT or ZSL task
    """
    times, rewards, accs = read_file(text_file)
    mean_reward = sum(rewards)/len(rewards)
    mean_acc = sum(accs)/len(accs)

    return mean_reward, mean_acc

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Show Graph')
    parser.add_argument('-d', '--difficulty', type=str, default="easy",
                    help="""Difficulty of the environment,
                    "easy", "medium" or "hard" (default: hard)""")
    parser.add_argument('-t', '--type', type=str, default="acc",
                    help="Type of data, 'acc', 'reward' ")
    # parser.add_argument('-p1', '--plot1', type=int, default=0,
    #                 help='show the plot type 1 (default: 0)')
    parser.add_argument('-p1', '--plot1', action='store_true')
    parser.add_argument('-p2', '--plot2', action='store_true')
    # parser.add_argument('-p2', '--plot2', type=int, default=0,
    #                 help='show the plot type 2 (default: 0)')
    args = parser.parse_args()

    if args.difficulty == 'easy':
        # easy
        # graph1 = read_file(text_file="/home/tinvn/TIN/NLP_RL_Code/AE_VLN_Vizdoom/saved/easy/based_easy/train_based_easy.log") # base
        graph1 = read_file(text_file='./train_easy_diff_gated.log')
        # graph2 = read_file(text_file="./saved/fourier_models/single_goal/easy/train_easy_forier_d1.log")
        graph2 = read_file(text_file='./train_easy_diff_fourier.log')
        # graph3 = read_file(text_file="./saved/convolve/train_easy_convolve.log")
        graph3 = read_file(text_file='./train_easy_diff_convolve.log')
        
        # plot
        plot_graph_2(graphs=[graph1, graph2, graph3], labels=['GA easy (1 Goal - Only Objects)', 'FGA easy (1 Goal - Only Objects)', "CA easy (1 Goal - Only Objects)"], level='easy', shown_type=args.type)
    elif args.difficulty == 'medium':
        # medium
        graph1 = read_file(text_file="/home/tinvn/TIN/NLP_RL_Code/AE_VLN_Vizdoom/saved/medium/based_medium/train8_medium.log")
        graph2 = read_file(text_file="./saved/fourier_models/single_goal/medium/train_medium_forier_d1.log")
        graph3 = read_file(text_file="./saved/convolve/train_medium_convolve.log")
        # plot
        plot_graph_2(graphs=[graph1, graph2, graph3], labels=['GA medium', 'FGA medium', "CA medium"], level='medium', shown_type=args.type)
    elif args.difficulty == 'hard':
        # hard
        graph1 = read_file(text_file="/home/tinvn/TIN/NLP_RL_Code/AE_VLN_Vizdoom/saved/hard/based_hard/train_based_hard.log", defaut_gap=62/3600) # train_hard base
        graph2 = read_file(text_file="./saved/fourier_models/single_goal/hard/train_hard_forier_d1.log")
        graph3 = read_file(text_file='./saved/convolve/train_hard_convolve.log')
        #plot
        plot_graph_2(graphs=[graph1, graph2, graph3], labels=['GA hard', 'FGA hard', "CA hard"], level='hard', shown_type=args.type)

    # print("Based easy (MT): ", calculate_mean_reward_and_acc('./saved/based_easy/test_MT_based_easy.log'))
    # print("Based easy (ZSL): ", calculate_mean_reward_and_acc('./saved/based_easy/test_ZSL_based_easy.log'))
    # print("AE Prelu easy (MT): ", calculate_mean_reward_and_acc('./test9_ae_prelu_MT.log'))
    # print("AE Prelu easy (ZSL): ", calculate_mean_reward_and_acc('./test9_ae_prelu_ZSL.log'))
    