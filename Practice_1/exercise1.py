import pandas as pd
import numpy as np
df = pd.read_csv('T.csv')
print(df)


print(df.head())

df.info()
print(len(df))

print(df.size)

print("Mean :       ", df.mean())
print("Standard deviation:", df.std())
print("Minimum :    ", df.min())
print("Maximum :    ", df.max())
print("mode :    ", df.mode())




import scipy.stats


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

print('the 95% confidence interval = ',mean_confidence_interval(df))



a=df['Seismic activity recorded over a 7 day period']
a=list(a)

samples = sorted(a)

def find_median(sorted_list):
    indices = []

    list_size = len(sorted_list)
    median = 0

    if list_size % 2 == 0:
        indices.append(int(list_size / 2) - 1)  # -1 because index starts from 0
        indices.append(int(list_size / 2))

        median = (sorted_list[indices[0]] + sorted_list[indices[1]]) / 2
        pass
    else:
        indices.append(int(list_size / 2))

        median = sorted_list[indices[0]]
        pass

    return median, indices
    pass

median, median_indices = find_median(samples)
Q1, Q1_indices = find_median(samples[:median_indices[0]])
Q2, Q2_indices = find_median(samples[median_indices[-1] + 1:])

quartiles = [Q1, median, Q2]

print("the first and third quartile, and the IQR = (Q1, median, Q3): {}".format(quartiles))

# Import libraries
import matplotlib.pyplot as plt

# Creating dataset
np.random.seed(10)
data = a

fig = plt.figure(figsize=(5, 5))

# Creating plot
plt.boxplot(data)

# show plot
plt.show()
