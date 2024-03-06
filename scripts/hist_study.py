import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("perceptive_study/quest-quill_answer.tsv", sep="\t")

df1 = df[df["quest_id"] == 2][["coherence", "information", "has_played"]]
df2 = df[df["quest_id"] == 4][["coherence", "information", "has_played"]]

plt.figure()
sns.set(color_codes=True)
sns.set(style="white", palette="bright")
sns.histplot(data=df2, x="coherence", hue="has_played", discrete=True)
plt.title("Coherence score for quest n°2 (generated)")
plt.savefig("perceptive_study/coherence_quest2.png")
plt.show()
