from glob import glob
import pandas as pd

lines=[]
for fn in glob("relevant_logs/*"):
    with open(fn,"r") as f:
        s = f.read()
        header_lines = s.split("="*50)[0].split('\n')
        footer_lines = s.split("-"*50)[-1].split('\n')
        top_p = float(header_lines[-4].split(':')[-1])
        top_k = float(header_lines[-3].split(':')[-1])
        temp = float(header_lines[-2].split(':')[-1])
        precision = float(footer_lines[-5].split(':')[-1])
        recall = float(footer_lines[-4].split(':')[-1])
        f1 = float(footer_lines[-3].split(':')[-1])
        lines.append({'top_p':top_p, 'top_k':top_k, 'temperature':temp, 
                      'precision':precision, 'recall':recall, 'f1':f1})

df = pd.DataFrame(lines)
df.sort_values(by='f1',ascending=False, inplace=True)
print(df)
