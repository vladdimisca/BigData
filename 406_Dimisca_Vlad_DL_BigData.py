import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report


stroke_file = "Data/stroke-data.csv"

print("READING FROM THE INPUT FILE\n")

# read the data from the csv file
df = pd.read_csv(stroke_file)

print("DISPLAY THE FIRST ROWS FROM THE DATAFRAME\n")
print(df.head())

print("\nCHECK IF THERE ARE NULL VALUES\n")
print(df.isnull().sum())

print("\nDISPLAY DESCRIPTIVE STATISTICS\n")
print(df.describe().transpose())

# plot the distribution of the column 'age'
plt.figure(figsize=(12, 5))
sns.histplot(df['age'], kde=True)
plt.show()

# plot a countplot diagram with stroke/not stroke numbers
sns.countplot(x='stroke', data=df)
plt.show()

print("\nCLEANING THE DATA\n")

# drop irrelevant columns ('id')
df = df.drop("id", axis=1)

# cast columns containing numeric values (but provided as string) to double ('bmi')
df["bmi"] = df["bmi"].astype(float)

# remove the rows where bmi is null and smoking_status is unknown
df = df[(df["bmi"].notnull()) | (df["smoking_status"] != "Unknown")]

# if the bmi value is unknown, replace it with the mean value from that column
df["bmi"] = df["bmi"].fillna(df["bmi"].mean())

# transform relevant columns containing string values into categorical variables
columns_to_transform = [
    "gender",
    "ever_married",
    "work_type",
    "Residence_type",
    "smoking_status"
]

df[columns_to_transform] = df[columns_to_transform].apply(lambda x: x.astype("category").cat.codes)

# separate the characteristics of the target variable
X = df.drop("stroke", axis=1)
y = df["stroke"]

# divide the dataset into train and test subsets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

# apply a feature scaling on all variables
scaler = MinMaxScaler()

scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

print("----CLASSIFICATION USING SEQUENTIAL MODEL----\n")

model = Sequential()
model.add(Dense(units=30, activation='relu', input_shape=(10,)))  # first layer
model.add(Dropout(0.6))  # used to avoid overfitting (input layer - best between 0.5 - 0.8)
model.add(Dense(units=30, activation='relu'))  # second layer
model.add(Dropout(0.5))  # used to avoid overfitting (hidden layer - best ~0.5)
model.add(Dense(units=1, activation='sigmoid'))  # output layer - use sigmoid because it is a binary classification
print(model.summary())

model.compile(optimizer='nadam', loss='binary_crossentropy', metrics=['accuracy'])

# use EarlyStopping to avoid overfitting
early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25)

history = model.fit(x=X_train,
                    y=y_train,
                    epochs=150,
                    validation_data=(X_test, y_test),
                    verbose=1,
                    callbacks=[early_stop]
                    )

print("\nDISPLAY THE CLASSIFICATION REPORT\n")

y_pred = (model.predict(X_test) > 0.5).reshape((-1,))
print(classification_report(y_test, y_pred))
