import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import seaborn as sns
import pandas as pd

def checkdata(fpath,show_plot = True):
    # fpath = "data/moses/prop_temp.csv"
    results = pd.read_csv(fpath)

    logP, tPSA, QED = results.iloc[:, 0], results.iloc[:, 1], results.iloc[:, 2]

    figure, ((ax1,ax2,ax3)) = plt.subplots(nrows=1, ncols=3)

    sns.violinplot(y = "logP", data =results, ax=ax1, color=sns.color_palette()[0])
    sns.violinplot(y = "tPSA", data =results, ax=ax2, color=sns.color_palette()[1])
    sns.violinplot(y = "QED", data =results, ax=ax3, color=sns.color_palette()[2])

    ax1.set(xlabel='logP', ylabel='')
    ax2.set(xlabel='tPSA', ylabel='')
    ax3.set(xlabel='QED', ylabel='')

    bound_logP = get_quatiles(logP)
    for i in range(4):
        text = ax1.text(0, bound_logP[i], f'{bound_logP[i]:.2f}', ha='right', va='center', fontweight='bold', size=10, color='white')
        text.set_path_effects([path_effects.Stroke(linewidth=3, foreground='black'), path_effects.Normal(), ])

    bound_tPSA = get_quatiles(tPSA)
    for i in range(4):
        text = ax2.text(0, bound_tPSA[i], f'{bound_tPSA[i]:.2f}', ha='right', va='center', fontweight='bold', size=10, color='white')
        text.set_path_effects([path_effects.Stroke(linewidth=3, foreground='black'), path_effects.Normal(), ])

    bound_QED = get_quatiles(QED)
    for i in range(4):
        text = ax3.text(0, bound_QED[i], f'{bound_QED[i]:.2f}', ha='right', va='center', fontweight='bold', size=10, color='white')
        text.set_path_effects([path_effects.Stroke(linewidth=3, foreground='black'), path_effects.Normal(), ])
    if show_plot:
        plt.show()

    logP_max, logP_min = min(bound_logP[0], logP.max()), max(bound_logP[-1], logP.min())
    tPSA_max, tPSA_min = min(bound_tPSA[0], tPSA.max()), max(bound_tPSA[-1], tPSA.min())
    QED_max, QED_min = min(bound_QED[0], QED.max()), max(bound_QED[-1], QED.min())

    return logP_max, logP_min, tPSA_max, tPSA_min, QED_max, QED_min


def get_quatiles(df):
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    UAV = Q3 + 1.5 * IQR
    LAV = Q1 - 1.5 * IQR
    return [UAV, Q3, Q1, LAV]