import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

var_list = ['OUTCOME','AGE','WORKCLASS','FNLWGT','EDUCATION-NUM','RACE',
            'SEX','HOURS-PER-WEEK']

categorical_list = ['WORKCLASS','RACE']

work_dict = {' ?':'OTHER',' Never-worked':'OTHER',' Without-pay':'OTHER',
            ' Federal-gov':'PUB',' State-gov':'PUB',' Local-gov':'PUB',
            ' Private':'PRIVATE',' Self-emp-inc':'PRIVATE',
            ' Self-emp-not-inc':'PRIVATE'}


def main():
    df = pd.read_csv("adult.csv")
    df = df[var_list]

    #Combine Work sectors into public, self, other, and private
    df['WORKCLASS'] = df['WORKCLASS'].map(work_dict)

    # One hot encode categorical variables
    for var_name in categorical_list:
        df[var_name] = pd.Categorical(df[var_name])
        df_dummies = pd.get_dummies(df[var_name], prefix = var_name)
        df = pd.concat([df, df_dummies], axis=1)
        df = df.drop(var_name, 1)

    # Convert strings to binary variables
    df['SEX'] = (df['SEX'] == ' Male').astype(int)
    df['OUTCOME'] = (df['OUTCOME'] == " >50K").astype(int)

    df = (df-df.min())/(df.max()-df.min())

    train, test = train_test_split(df, test_size=0.2)

    train.to_csv("adult_numeric_train.csv", index=False)
    test.to_csv("adult_numeric_test.csv", index=False)


if __name__ == '__main__':
    main()
