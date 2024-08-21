import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.io import arff
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

data = arff.loadarff('Rice_Cammeo_Osmancik.arff')
df = pd.DataFrame(data[0])

encoder = LabelEncoder()
df['Class'] = encoder.fit_transform(df['Class'])

x = df.drop('Class', axis=1)
y = df['Class']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

model = LogisticRegression()
model.fit(x_train, y_train)

score = model.score(x_test, y_test)