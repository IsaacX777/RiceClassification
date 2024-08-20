import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.io import arff
from sklearn.linear_model import LinearRegression

data = arff.loadarff('Rice_Cammeo_Osmancik.arff')
df = pd.DataFrame(data[0])

y = df['Class']
x = df.drop('Class', axis=1)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

model = LinearRegression()
model.fit(x_train, y_train)