import pandas as pd
import numpy as np
import pickle
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest,chi2
from sklearn.impute import SimpleImputer
#load the csv file

df = pd.read_csv('C:\\Users\\Bhushan\\Desktop\\botnet project\\Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv')

print(df.head())
'''
#Select independent and dependent variable
X = df[[ ' selected_features']]
y = df[[' Label']]
print(X)

#Split the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

#feature scaling 
encoder = LabelEncoder()
X_train = encoder.fit_transform(X_train)
X_test = encoder.fit_transform(X_test)

#Instantiate the model
classifier = RandomForestClassifier()

#Fit the model
classifier.fit(X_train, y_train)

#make pickle file of our model
pickle.dump(classifier, open("model.pkl", "wb"))'''

Target= ' Label'
X = df.loc[:,df.columns !=Target]
y = df.loc[:,Target]

encoder = LabelEncoder()
df[' Label']=encoder.fit_transform(df[' Label'])

#

lower_bound=0
upper_bound=100
X_clipped = np.clip(X, lower_bound, upper_bound) 
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X_clipped)
#
selector = SelectKBest(score_func=chi2, k=10)
selector.fit(X_imputed, y)
selected_features = selector.get_support(indices=True)
X_new = X_imputed[:, selected_features]
X_new_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.25)
selected_features=['Destination Port', ' Flow Duration', ' Total Fwd Packets',
       ' Total Backward Packets', 'Total Length of Fwd Packets',
       ' Total Length of Bwd Packets', ' Fwd Packet Length Max',
       ' Fwd Packet Length Min', ' Fwd Packet Length Mean',
       ' Fwd Packet Length Std']
X = df[selected_features]
print(X)

#
rf_reg= RandomForestClassifier(n_estimators=500 )
rf_reg.fit(X_new_train,y_train)

#
model = RandomForestClassifier(max_depth=10)  # Set max_depth to the desired value, e.g., 10

# Train the model on your training data
model.fit(X_new_train, y_train)

joblib.dump(model, open("model.pkl", "wb"))