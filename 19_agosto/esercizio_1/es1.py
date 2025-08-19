import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Carica il file CSV
df = pd.read_csv("C:\\Users\\KB316GR\\OneDrive - EY\\Desktop\\Dataset\\AEP_hourly.csv", parse_dates=['Datetime'])

print(df.shape[0])

# Estraggo la data per calcolare la media giornaliera
df['Date'] = df['Datetime'].dt.date

# Calcolo la media giornaliera
daily_mean = df.groupby('Date')['AEP_MW'].transform('mean')

# Creo la colonna target: 1 = ALTO CONSUMO, 0 = BASSO CONSUMO
df['High_Consumption'] = (df['AEP_MW'] > daily_mean).astype(int)

df['Hour'] = df['Datetime'].dt.hour
df['Weekday'] = df['Datetime'].dt.weekday
df['Month'] = df['Datetime'].dt.month

print(df.shape[0])

X = df[['Hour', 'Weekday', 'Month']]
y = df['High_Consumption']

# Suddivido in training e test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))


plt.figure(figsize=(5, 5))
plot_tree(clf, feature_names=X.columns, class_names=["Basso", "Alto"], filled=True)
plt.title("Albero Decisionale per Classificazione del Consumo Energetico")
plt.show()
