import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Read in the data
df = pd.read_csv('results - Feuille 1.tsv', sep='\t', names=["corpus", "value", "metric", "type", "technique"])
#get lines where type is "partial" and "metric" ends with "_f1"
df = df[(df.type == "partial") & (df.metric.str.endswith("_f1"))]

metrics = df.metric.unique()

grid = sns.FacetGrid(df, row="corpus")
grid.map_dataframe(sns.barplot, x="technique", y="value", hue="metric", hue_order=metrics, palette="deep")
#sns.barplot(data=df, x="technique", y="value", hue="metric", hue_order=metrics)
plt.show()