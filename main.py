import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


data = pd.read_csv('WINE.csv')

if data.isnull().values.any():
    data.fillna(data.mean())

print(data.info())
print(data.describe(include='all'))

x = data.iloc[:, 2:]
y = data.iloc[:, 1]

x.iloc[:, 1:].head()

x.plot(kind='box', subplots=True, layout=(9, 2), figsize=(10, 40))

from sklearn.covariance import EllipticEnvelope


ellenv = EllipticEnvelope(contamination=0.1, random_state=1)

# Returns 1 of inliers, -1 for outliers, we skip first string column
pred = ellenv.fit_predict(x.iloc[:, 1:])

outlier_index = np.where(pred == -1)
print(outlier_index)
outlier_values = x[outlier_index]

sns.scatterplot(x=x[:, 0], y=x[:, 1])
sns.scatterplot(x=outlier_values[:,0], 
                y=outlier_values[:,1], color='r')
plt.title("Elliptic Envelope Outlier Detection", fontsize=15, pad=15)
plt.show()
