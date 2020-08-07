import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn import preprocessing



#reading the dataset
df=pd.read_excel('titanic.xls')
#print(df.head())
#dropping the body and name column because they are irrelevant (all too varrying)
df.drop(['body', 'name'], 1, inplace=True)
#filling empty cells with 0
df.fillna(0, inplace=True)

#converting text data to numerical data
def handle_non_numerical_data (df):
    columns=df.columns.values
    
    for column in columns:
        text_digit_vals={}
        def convert_to_int(val):
            return text_digit_vals[val]
        
        if df[column].dtype !=np.int64 and df[column].dtype!=np.float64:
            column_contents=df[column].values.tolist()
            unique_elements=set(column_contents)
            x=0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique]=x
                    x+=1
                    
            df[column]=list(map(convert_to_int, df[column]))
    return df


df=handle_non_numerical_data(df)


#drpping the survived column to have other values combined as x
#preprocessing the dataset and creating the y values as the entire survived column
#splitting the dataset to 80%(train), 20%(test)
x=np.array(df.drop(['survived'], 1))
x=preprocessing.scale(x)
y=np.array(df['survived'])
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2)

#applying the svm algorithm 
clf=SVC(gamma='auto')
clf=clf.fit(x_train, y_train)
accuracy=clf.score(x_test, y_test)
print(accuracy)
