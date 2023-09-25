import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

#plot 20 grid lines instead of 5
sns.set_style("whitegrid")

# Read in the data
df = pd.read_csv('results - Feuille 1.tsv', sep='\t', header=0)
#get lines where type is "partial" and "metric" ends with "_f1"
df = df[(df.metric_type == "exact") & (df.metric.str.endswith("_f1"))]
print(df)

metrics = df.metric.unique()

grid = sns.FacetGrid(df, row="corpus")
grid.map_dataframe(sns.barplot, x="technique", y="value", hue="metric", hue_order=metrics, palette="deep")
grid.add_legend()
grid.fig.tight_layout()
grid.set(ylim=(0, 1))
left_margin = 0.05
plt.subplots_adjust(left=left_margin)
plt.show()