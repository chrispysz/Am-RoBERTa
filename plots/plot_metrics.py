import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
plt.close("all")

data_type = "len_var"

def plot_metric(metric, axs, a_x, a_y):
    path = "plots/"+data_type+"_"+metric+".csv"
    path_robust = "plots/"+data_type+"_"+metric+"_robust.csv"

    df = pd.read_csv(path)

    df['mean'] = df.iloc[:, 1:22].mean(axis = 1)
    df['min'] = df.iloc[:, 1:22].min(axis = 1)
    df['max'] = df.iloc[:, 1:22].max(axis = 1)
    df['average_mean'] = df.loc[df.index[-1], "mean"].mean()

    df.to_csv(path_robust, index=False)
    x = df["Step"]

    y_mean = df['mean']
    y_min = df['min']
    y_max = df['max']

    axs[a_x,a_y].plot(x, y_mean, label='mean', color = 'red', linewidth = 0.3)
    axs[a_x,a_y].plot(x, y_min, label='min, max', color = 'blue', linewidth = 0.3)
    axs[a_x,a_y].plot(x, y_max, color = 'blue', linewidth = 0.3)
    axs[a_x,a_y].grid(True)
    
    if metric=="f1":
        axs[a_x,a_y].set_title('f1 score')
    if metric=="precision":
        axs[a_x,a_y].set_title(metric)
    if metric=="recall":
        axs[a_x,a_y].set_title(metric)
    if metric=="accuracy":
        axs[a_x,a_y].set_title(metric)
    if metric=="fpr":
        axs[a_x,a_y].set_yscale('log')
        axs[a_x,a_y].set_title('false positive rate (log scale)')

fig, axs = plt.subplots(3, 2, sharex = False,sharey=False)
fig.suptitle('Metrics')
fig.tight_layout()
plot_metric("accuracy", axs, 0, 0)
plot_metric("f1", axs, 0,1)
plot_metric("fpr", axs,1,0)
plot_metric("precision", axs,1,1)
plot_metric("recall", axs,2,0)
fig.delaxes(axs[2][1])
plt.show()