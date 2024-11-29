""" Plot the metrics for various SC weights """

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from IPython.display import display
import math
import matplotlib.pylab as pylab

# plot params
params = {'legend.fontsize': 20,
         'axes.labelsize': 20,
         'axes.titlesize':24,
         'xtick.labelsize':20,
         'ytick.labelsize':20}
pylab.rcParams.update(params)

scaling = {"EMG_flex": (-50,100), "EMG_ext": (-50,100), "EMG_mean": (-50,100), 'rmse': (0,80), 'peaks': (0,15), 'peaks_first': (0,15), 'peaks_second': (0,15),'damping' : (0,30), 'damping_first' : (0,30), 'damping_second' : (0,30),
           'sal': (-4.5, -3), 'sparc': (-4, -2), 'log_jerk': (-15, -10), 'speed_arc_length': (-10, -2),
           'range_pos' : (0,150), 'range_pos_first' : (0,100), 'range_pos_second' : (1,2), 'range_speed' : (5,12), 'range_speed_first' : (5,12), 'range_speed_second' : (5,12)}
scaling_perturb = {'rmse': (0,80), 'deviation': (0,80), 'peaks': (0,10), 'peaks_first': (0,10), 'peaks_second': (0,15), 'damping' : (0,50), 'damping_first' : (0,10), 'damping_second' : (0,30),
           'sal': (-7, -2), 'sparc': (-6, -1), 'log_jerk': (-20, -5), 'speed_arc_length': (-6, -2),
           'range_pos' : (0,150), 'range_pos_first' : (0,100), 'range_pos_second' : (1,2), 'range_speed' : (5,12), 'range_speed_first' : (5,12), 'range_speed_second' : (5,12)}
metric_units = {"EMG_flex": "%", 'rmse': "°", 'deviation': "°", 'range_pos': "°"}


def main():

    # set scenario
    traj = "flexion"  # "flexion" or "circle"
    case = "movement"  #"movement" or "perturbation"
    pairs = True 
    joint = "elbow"

    if case == "movement":
        m_columns = ['rmse', 'speed_arc_length', "group", "weight", "control"] #"EMG_flex", "EMG_ext", "EMG_mean",
    elif case == "perturbation":
        m_columns = ['rmse', 'deviation', 'speed_arc_length', 'damping', "group", "weight", "control"] 
        
    # folder
    if not pairs:
        files = ["results/"+traj+"/"+case+"/control_input_nosc/fixed_/"]
        groups = ['Ia_In', 'Ia_Mn', 'Ib', 'II', 'Ia_Mn_syn', 'Rn_Mn'] 
            
        data = extract_metrics(files, joint, control='1')

        # plots
        metrics = select_groups_and_metrics(data, groups, m_columns)
        save_folder = files[0]
        if case == "perturbation":
            lineplots(metrics, figsize=(20,8), m_columns=m_columns, save_folder=save_folder, control='1', scaling=scaling_perturb)
        else:
            lineplots(metrics, figsize=(20,8), m_columns=m_columns, save_folder=save_folder, control='1')

    else:
        pairs = ['Ia_Mn','Ia_In']
        files = ["results/"+traj+"/"+case+"/control_input_nosc/pair_"+pairs[0]+"_"+pairs[1]+"/"]
        data = extract_metrics_pairs(files, joint, case)

        # plots
        figsize=figsize=(20,12)
        save_folder = files[0]
        heatmap_pairs(data, m_columns, figsize, pairs, case, save_folder)

    plt.show()


def extract_metrics(files, joint, control='1'):
    
    for file in files:
        elbow_ = pd.read_excel(file + joint+"_metric2s.xlsx")
        elbow_["weight"] = elbow_.groupby("group",  group_keys=False)["Unnamed: 0"].apply(lambda x: x.str.split("_", expand=True).iloc[:, -1])
        elbow_["weight"] = pd.to_numeric(elbow_["weight"]).round(1)
        elbow_["control"] = control
        if file != files[0]:
            elbow = pd.concat([elbow_, elbow], axis=0)
        else :
            elbow = elbow_
    return elbow
    

def extract_metrics_pairs(files, joint, type):

    for file in files:
        elbow_ = pd.read_excel(file + joint+ "_metric2s.xlsx")
        elbow_["group1"] = elbow_["pair1"].str.rsplit('_', 1).str[0]
        elbow_["group2"] = elbow_["pair2"].str.rsplit('_', 1).str[0]
        elbow_["weight1"] = elbow_["pair1"].str.rsplit('_', 1).str[1]
        elbow_["weight1"] = pd.to_numeric(elbow_["weight1"]).round(1)
        elbow_["weight2"] = elbow_["pair2"].str.rsplit('_', 1).str[1]
        elbow_["weight2"] = pd.to_numeric(elbow_["weight2"]).round(1)
        elbow_["type"] = type
        if file != files[0]:
            elbow = pd.concat([elbow_, elbow], axis=0)
        else :
            elbow = elbow_
    return elbow


def select_groups_and_metrics(elbow, groups, m_columns):

    #Select metrics
    elbow_data = elbow[m_columns]
    #Select groups
    elbow_metrics = elbow_data[elbow_data['group'].isin(groups)]
    
    return elbow_metrics


def lineplots(elbow_metrics, figsize, m_columns, save_folder, control=1, set_y_lim = True, scaling=scaling, metric_units=metric_units):
    
    elbow_metrics_ = elbow_metrics[elbow_metrics["control"]==control]
    if len(m_columns[:-3]) > 6 :
        fig1, axes1 = plt.subplots(nrows=2, ncols=3, figsize=figsize) 
        fig2, axes2 = plt.subplots(nrows=(len(m_columns[:-3])-6)//3+1, ncols=3, figsize=(20,6))
        for idx, name in enumerate(m_columns[:-3]):
            if idx < 6:
                row = idx // 3
                col = idx % 3
                ax = axes1[row, col]
            else:
                row = (idx-6) // 3
                col = (idx-6) % 3
                if row == 0:
                    ax = axes2[col]
                else:
                    ax = axes2[row, col]
            if name.split("_first")[0].split("_second")[0] in list(metric_units.keys()):
                if metric_units[name.split("_first")[0].split("_second")[0]] == "°":
                    elbow_metrics_[name] = elbow_metrics_[name]*180/math.pi
            if col + row == 0:
                sns.lineplot(data=elbow_metrics_,x=elbow_metrics_["weight"],y=elbow_metrics_[name], hue = "group", ax=ax)
            else:
                sns.lineplot(data=elbow_metrics_,x=elbow_metrics_["weight"],y=elbow_metrics_[name],  hue = "group", ax=ax, legend=False)
            ax.set_title(name)
            if row == 1:
                ax.set_xlabel('weight')
            else:
                ax.set_xlabel('')
            if name.split("_first")[0].split("_second")[0] in list(metric_units.keys()):
                ax.set_ylabel(name + " ("+metric_units[name.split("_first")[0].split("_second")[0]]+")")
            else:
                ax.set_ylabel(name)
            ax.spines[['right', 'top']].set_visible(False)
            if set_y_lim :
                ax.set_ylim(scaling[name.split("_second")[0]])

        plt.tight_layout()
        fig1.savefig(save_folder + "plot_metrics")
        fig1.savefig(save_folder + "plot_metrics.svg")
        if len(m_columns) > 6:
            fig2.savefig(save_folder + "plot_metrics2")
            fig2.savefig(save_folder + "plot_metrics2.svg")

    else:
        if len(m_columns[:-3]) > 3 :
            fig, axes = plt.subplots(nrows=2, ncols=3, figsize=figsize)
        else:
            fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20,6))
        for idx, name in enumerate(m_columns[:-3]):
            if len(m_columns[:-3]) > 3 :
                row = idx // 3
                col = idx % 3
                ax = axes[row, col]
            else:
                row = 0
                col = idx % 3
                ax = axes[col]
            if name.split("_first")[0].split("_second")[0] in list(metric_units.keys()):
                if metric_units[name.split("_first")[0].split("_second")[0]] == "°":
                    elbow_metrics_[name] = elbow_metrics_[name]*180/math.pi
            if row == 0 and col == 0:
                sns.lineplot(data=elbow_metrics_,x=elbow_metrics_["weight"],y=elbow_metrics_[name], hue = "group", ax=ax)
            else:
                sns.lineplot(data=elbow_metrics_,x=elbow_metrics_["weight"],y=elbow_metrics_[name], hue = "group", ax=ax, legend=False)
            ax.set_title(name)
            ax.set_xlabel('weight')
            if name.split("_first")[0].split("_second")[0] in list(metric_units.keys()):
                ax.set_ylabel(name + " ("+metric_units[name.split("_first")[0].split("_second")[0]]+")")
            else:
                ax.set_ylabel(name)
            if set_y_lim :
                ax.set_ylim(scaling[name.split("_first")[0].split("_second")[0]])
            ax.spines[['right', 'top']].set_visible(False)

        plt.tight_layout()
        plt.savefig(save_folder + "plot_metrics")
        plt.savefig(save_folder + "plot_metrics.svg")
    

def lineplot_controls(elbow_metrics, figsize, controls, column_name, scale = None):
    
    fig, axes = plt.subplots(nrows=1, ncols=len(controls), figsize=figsize)
    for idx, control in enumerate(controls):
        elbow_metrics_ = elbow_metrics[elbow_metrics["control"]==control]
        col = idx % len(controls)
        ax = axes[col]
        sns.lineplot(data=elbow_metrics_,x=elbow_metrics_["weight"],y=elbow_metrics_[column_name], hue = "group", ax=ax)
        ax.set_title(column_name + " control" + control)
        ax.set_xlabel('')
        ax.set_ylabel(column_name + " control" + control)
        if "sal" not in column_name and "sparc" not in column_name and 'speed_arc_length' not in column_name:
            if scale is None :
                ax.set_ylim(scaling[column_name])
            else :
                ax.set_ylim(scale)
        else:
            if scale is not None :
                ax.set_ylim(scale)


def heatmap_pairs(elbow, m_columns, figsize, pair, type_, save_folder):
    
    # Create a 2D array to hold the result values corresponding to w0 and w1
    elbow_ = elbow[elbow["type"]== type_]
    elbow_ = elbow_[elbow_["group1"]==pair[0]]
    elbow_ = elbow_[elbow_["group2"]==pair[1]]
    #display(elbow_)

    grid_size = len(np.unique(elbow_["weight1"]))
    result_grid = np.zeros((grid_size, grid_size))
    
    fig, axes = plt.subplots(nrows=len(m_columns[:-3])//3+1, ncols=3, figsize=figsize)
    for idx, name in enumerate(m_columns[:-3]):
        row = idx // 3
        col = idx % 3
        if row > 1:
            ax = axes[row, col]
        else:
            ax = axes[col]
        if name in list(metric_units.keys()):
            if metric_units[name] == "°":
                elbow_[name] = elbow_[name]*180/math.pi
        for i, (x, y, val) in enumerate(zip(elbow_["weight1"], elbow_["weight2"], elbow_[name])):
            result_grid[x, y] = val
    
        #create heatmap
        heatmap = ax.pcolormesh(result_grid, cmap='viridis')
        # Add a colorbar
        cbar = fig.colorbar(heatmap, ax=ax)
        if name in list(metric_units.keys()):
            cbar.set_label(name+ " ("+metric_units[name]+")", rotation=270)
        else:
            cbar.set_label(name, rotation=270)
        # Set tick labels and positions
        ax.set_xticks(np.arange(grid_size), np.unique(elbow_["weight1"]))
        ax.set_yticks(np.arange(grid_size), np.unique(elbow_["weight1"]))
        ax.set_title(name)
        ax.set_xlabel(elbow_["group2"].iloc[0])
        ax.set_ylabel(elbow_["group1"].iloc[0])
        
    
        """# Display values within the heatmap cells
        for i in range(grid_size):
            for j in range(grid_size):
                ax.text(j + 0.5, i + 0.5, f'{result_grid[i, j].round(1):.2f}', ha='center', va='center', color='black', fontsize=7)"""
        
    plt.tight_layout()
    plt.savefig(save_folder + "heatmap_metrics")
    plt.savefig(save_folder + "heatmap_metrics.svg")
    

if __name__ == '__main__':
    main()
