import gzip
from collections import defaultdict
import random
import numpy as np
from sklearn.metrics import mean_squared_error
from dask.array.random import beta
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.tools.plotting import radviz

sns.set(style="white", color_codes=True)


    
path = "../amazon-fine-foods/"
# training_all=[]
# for l in readGz(path+"finefoods.txt.gz"):
#     training_all.append(l)
#     
# print "done"

food = pd.read_csv(path+"Reviews.csv",header=0)
food.sample(frac=1)

# food.head()
food["length_text"]=food["Text"].str.len()
print food["Score"].value_counts()
print food[:1]
earliest= food["Time"].min()
food["Time_compress"]=np.around((food["Time"]-earliest)*1.0/31536000)
# food = food[:1000]
# scorearray=food.as_matrix(columns=["Score"])
# sns.distplot(a=scorearray,bins=(0.5,1.5,2.5,3.5,4.5,5.5),norm_hist =False,kde=False)
# timeseries=food.groupby('Time_compress', as_index=False)['Score'].mean()
# timeseries=food['Time_compress'].value_counts()
# timeseries.sort()
print food["Score"].mean()

timeseries=food.groupby('ProductId', as_index=False)['Score'].mean()
# print timeseries
print len(timeseries.index)
sns.set_style("darkgrid", {'text.color': '0.25'})
# sns.distplot(a=timeseries["Score"],bins=(1,1.25,1.5,1.75,2,2.25,2.5,2.75,3,3.25,3.75,4,4.25,4.5,4.75,5),norm_hist =False,kde=False,color="red")
sns.boxplot(x="Time_compress", y="Score",  data=food)
# x=timeseries["Time_compress"]
# y=timeseries["Score"]
# # sns.kdeplot(x, y,kernel={'gau'})
# # sns.regplot(x="Time_compress", y="Score", data=timeseries,scatter_kws={"s": 80},order=15, ci=None, truncate=True)
# timeseries.plot(kind="line")





# food = food[:200000]
# print food.Text
# food.plot(kind="scatter", x="Time", y="HelpfulnessNumerator")

# sns.jointplot(y="length_text", x="Score", data=food, size=5)
# sns.regplot(x="Time_compress", y="HelpfulnessDenominator", data=food, order=2, ci=None, truncate=True)
# food.plot(kind="scatter", x="Time_compress", y="HelpfulnessDenominator")
# sns.FacetGrid(food, hue="Score", size=5) \
#    .map(plt.scatter, "length_text", "HelpfulnessNumerator") \
#    .add_legend()
# helpfulness=food.query("HelpfulnessDenominator != 0")
# sns.boxplot(y="HelpfulnessDenominator", x="Time_compress", data=helpfulness)

# sns.boxplot(x="Score", y="HelpfulnessNumerator", data=helpfulness)


# x=food["Time"]
# sns.FacetGrid(food, hue="Score", size=5) \
#    .map(plt.scatter, "Time", "HelpfulnessNumerator") \
#    .add_legend()
# sns.distplot(x)

# plt.xlim(0,6000)
# plt.ylim(3.5,5)
plt.title('Distribution of Average Rating received by 74258 items')
plt.xlabel('Average Review Rating')
plt.show()
