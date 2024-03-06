import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("perceptive_study/quest-quill_answer.tsv", sep="\t")

df1 = df[df["quest_id"] == 2][["coherence", "information", "has_played"]]
df2 = df[df["quest_id"] == 4][["coherence", "information", "has_played"]]

plt.figure()
sns.set(color_codes=True)
sns.set(style="white", palette="bright")
sns.histplot(data=df2, x="information", hue="has_played", discrete=True)
plt.title("Information score for quest n°4 (generated)")
plt.savefig("perceptive_study/info_quest4.png")
plt.show()
