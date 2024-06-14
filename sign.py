import numpy as np
import pandas as pd
from deepsig import aso

# Load data
filename = "res.ods"
df = pd.read_excel(filename, engine="odf", header=0)
print(df)
seed = 666
np.random.seed(seed)

colnames = df.columns
for colname in colnames:
    if "Unnamed" in colname:
        continue
    all_values = [float(x) for x in df[colname].values if x != '-']
    CLM_values = all_values[:13]
    MLM_values = all_values[14:]
    print(colname)
    # print("CLM: ", CLM_values)
    # print("MLM: ", MLM_values)

    min_eps = aso(MLM_values, CLM_values, seed=seed, show_progress=False, confidence_level=0.95)
    if min_eps < 0.2:
        print('MLM is clearly better than CLM, eps =', round(min_eps, 3))
        # print('min_eps =', min_eps)
    elif min_eps < 0.4:
        print("MLM is probably better than CLM, eps =", round(min_eps, 3))
        # print('min_eps =', min_eps)
    else:
        print("MLM is boring..., eps =", round(min_eps, 3))
        # print('min_eps =', min_eps)