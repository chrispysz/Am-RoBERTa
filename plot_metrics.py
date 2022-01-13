import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
plt.close("all")


def plot_metric(metric, axs, a_x, a_y):
    path = "plots/"+metric+".csv"
    path_robust = "plots/"+metric+"_robust.csv"

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
        axs[a_x,a_y].set_title('F1 score')
        axs[a_x,a_y].set_yticks(np.arange(0,1.2,0.2))
    if metric=="precision":
        axs[a_x,a_y].set_title("Precision")
        axs[a_x,a_y].set_yticks(np.arange(0,1.2,0.2))
    if metric=="recall":
        axs[a_x,a_y].set_title("Recall")
        axs[a_x,a_y].set_yticks(np.arange(0,1.2,0.2))
    if metric=="accuracy":
        axs[a_x,a_y].set_title("Accuracy")
    if metric=="fpr":
        axs[a_x,a_y].set_yscale('log')
        axs[a_x,a_y].set_title('False positive rate (log scale)')

fig, axs = plt.subplots(3, 2, constrained_layout=True)
fig.suptitle('PDB_len21')
fig.tight_layout()
plot_metric("accuracy", axs, 0, 0)
plot_metric("f1", axs, 0,1)
plot_metric("fpr", axs,1,0)
plot_metric("precision", axs,1,1)
plot_metric("recall", axs,2,0)
fig.delaxes(axs[2][1])
plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.3, 
                    hspace=0.7)
plt.show()