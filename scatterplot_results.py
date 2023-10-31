from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns

B=10**9
M=10**6

dummy = [
    {
        "model_name":"BLOOM",
        "model_type":"Causal",
        "model_size": 176*B,
        "model_domain": "General",
        "general_performance": 0.7,
        "medical_performance": 0.6,
    },
    {
        "model_name":"Vicuna-13b",
        "model_type":"Causal",
        "model_size": 13*B,
        "model_domain": "General",
        "general_performance": 0.8,
        "medical_performance": 0.5,
    },
    {
        "model_name":"Mistral-7b",
        "model_type":"Causal",
        "model_size": 7*B,
        "model_domain": "General",
        "general_performance": 0.85,
        "medical_performance": 0.4,
    },
    {
        "model_name":"BERT-base",
        "model_type":"Masked",
        "model_size": 110*M,
        "model_domain": "General",
        "general_performance": 0.85,
        "medical_performance": 0.7,        
    },
    {
        "model_name":"BioBERT",
        "model_type":"Masked",
        "model_size": 110*M,
        "model_domain": "Medical",
        "general_performance": 0.5,
        "medical_performance": 0.8,
    },
    {
        "model_name":"Dr-BERT",
        "model_type":"Masked",
        "model_size": 110*M,
        "model_domain": "Medical",
        "general_performance": 0.6,
        "medical_performance": 0.7,
    },
    {
        "model_name":"BERT-large",
        "model_type":"Masked",
        "model_size": 340*M,
        "model_domain": "General",
        "general_performance": 0.9,
        "medical_performance": 0.75,
    },
    {
        "model_name":"Falcon-40b",
        "model_type":"Causal",
        "model_size": 40*B,
        "model_domain": "General",
        "general_performance": 0.9,
        "medical_performance": 0.6,
    },
]

#convert to pandas dataframe
dummy = pd.DataFrame(dummy)

sns.scatterplot(
    x="general_performance",
    y="medical_performance",
    hue="model_domain",
    hue_order=["General","", "Medical"],
    size="model_size",
    style="model_type",
    style_order=["Causal","", "Masked"],
    sizes=(500,10000),
    data=dummy)

#add labels
for i in range(len(dummy)):
    plt.text(
        x=dummy.general_performance[i]+0.01,
        y=dummy.medical_performance[i]+0.01,
        s=dummy.model_name[i],
        fontdict=dict(color='black',size=10),
        bbox=dict(facecolor='white',alpha=0.5,edgecolor='black',boxstyle='round,pad=0.5')
    )

plt.title("General vs Medical NER Performance of Language Models")
plt.xlabel("General Performance")
plt.ylabel("Medical Performance")
plt.xlim(0,1)
plt.ylim(0,1)
plt.tight_layout()
plt.gca().set_aspect('equal', adjustable='box')

#make sure the big circles in the legend don't overlap by spacing them out
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., scatterpoints=1, labelspacing=3)


plt.show()