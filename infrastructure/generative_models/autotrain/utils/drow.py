import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("generative_models/transformer/autotrain/train_ALZHEIMER/ALZHEIMER/weights/History_128_epo=35_20250423.csv")
df = df.drop(columns=["Unnamed: 0"])  


plt.style.use('default')

num_cols = len(df.columns) - 1  
fig, axes = plt.subplots(nrows=num_cols, ncols=1, figsize=(10, 3 * num_cols), sharex=True)

for ax, col in zip(axes, df.columns[1:]): 
    ax.plot(df['epochs'], df[col], marker='o')
    ax.set_title(f'{col} over Epochs', fontsize=12)
    ax.set_ylabel(col)
    ax.grid(True)

axes[-1].set_xlabel('Epochs')

plt.tight_layout()
plt.show()