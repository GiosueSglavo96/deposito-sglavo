
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.tree import plot_tree
from datetime import datetime

from imblearn.over_sampling import SMOTE


# Caricamento e pulizia dati
df = pd.read_csv("C:\\Users\\KB316GR\\OneDrive - EY\\Desktop\\Dataset\\AirQualityUCI.csv", sep=';', decimal=',')
pollutant = 'CO(GT)'

#print(df.head())
#print("Lunghezza dataset: " + str(df.shape[0]))

df = df.dropna(how='all', axis=1).dropna(how='any')

df = df[df['CO(GT)'] != -200]

df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H.%M.%S')
df.set_index('Datetime', inplace=True)

# Aggiunta di nuove feature temporali
df['Hour'] = df.index.hour          
df['Weekday'] = df.index.weekday     
df['Month'] = df.index.month         

#print(df.head())
#print("Lunghezza dataset pulito: " + str(df.shape[0]))

# Selezione inquinante

# Classificazione qualità dell'aria
df['DailyMean'] = df[pollutant].resample('D').transform('mean')
df['WeeklyMean'] = df[pollutant].resample('W').transform('mean')
global_mean = df[pollutant].mean()

def classify(row):
    if row[pollutant] > row['DailyMean']:
        return 1  # Scarsa qualità
    else:
        return 0  # Buona qualità

df['AirQuality'] = df.apply(classify, axis=1)

print(df.head())

# Verifica dello sbilanciamento delle classi
class_counts = df['AirQuality'].value_counts()
print("Distribuzione delle classi:")
print(class_counts)

# Percentuali
percentuali = df['AirQuality'].value_counts(normalize=True) * 100
print("\nPercentuale per classe:\n")
print(percentuali)


# Identificazione ore di picco giornaliere
peak_hours = df.groupby(df.index.date)[pollutant].nlargest(3).reset_index()
peak_hours = peak_hours.rename(columns={'level_1': 'Datetime', pollutant: 'PeakValue'})


# Percentuale settimanale di scarsa qualità
weekly_stats = df.resample('W').apply({
    'AirQuality': lambda x: (x.sum() / len(x)) * 100
})
weekly_stats['GlobalMean'] = df['AirQuality'].mean() * 100

# Preparazione dati per Random Forest
features = ['PT08.S1(CO)', 'NMHC(GT)', 'C6H6(GT)', 'PT08.S2(NMHC)', 'NOx(GT)', 'PT08.S3(NOx)',
            'NO2(GT)', 'PT08.S4(NO2)', 'PT08.S5(O3)', 'T', 'RH', 'AH', 'Hour', 'Weekday', 'Month']
X = df[features]
y = df['AirQuality']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print("Distribuzione dopo SMOTE:")
print(pd.Series(y_train_resampled).value_counts())


rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
rf.fit(X_train_resampled, y_train_resampled)
y_pred = rf.predict(X_test)


# Report delle metriche
print("\nReport delle metriche di classificazione:\n")
print(classification_report(y_test, y_pred))

plt.figure(figsize=(10, 5))
plot_tree(rf.estimators_[0], feature_names=features, filled=True, rounded=True, max_depth=3)
plt.title("Visualizzazione del primo albero della Random Forest")
plt.show()

"""
# Visualizza le ore di picco
plt.figure(figsize=(12, 6))
for date in peak_hours['level_0'].unique():
    subset = peak_hours[peak_hours['level_0'] == date]
    times = df.iloc[subset['Datetime']].index.time
    plt.plot(times, subset['PeakValue'], marker='o', label=str(date))

plt.xlabel('Ora')
plt.ylabel(f'Valore di {pollutant}')
plt.title('Top 3 ore di picco giornaliere')
plt.legend()
"""