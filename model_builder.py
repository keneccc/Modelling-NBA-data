from sklearn.model_selection import TimeSeriesSplit

from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

from predict import *


rr= RidgeClassifier(alpha=1)
split= TimeSeriesSplit(n_splits=3)

sfs = SequentialFeatureSelector(rr,n_features_to_select=30,direction="forward",cv=split)


## Cleaning data 

removed_columns=["season", "date", "won", "target","team","team_opp"]

selected_columns=df.columns[~df.columns.isin(removed_columns)]

#Scaling data 
scaler= MinMaxScaler()
df[selected_columns]=scaler.fit_transform(df[selected_columns])

sfs.fit(df[selected_columns],df["target"])

predictors = list (selected_columns[sfs.get_support()])

def backtest(data, model, predictors,start=2,step=1):
    all_predictions=[]

    seasons=sorted(data["season"].unique())
    for i in range(start,len(seasons), step):
        season=seasons[i]

        train=data[data["season"]< season]
        test= data[data["season"]== season]
        
        model.fit(train[predictors],train['target'])

        preds= model.predict(test[predictors])
        preds= pd.Series(preds, index=test.index)

        combined =pd.concat([test["target"], preds], axis =1)
        combined.columns= ["actual", "prediction"]

        all_predictions.append(combined)
    return pd.concat(all_predictions)



if __name__=="__main__":
    predictions = backtest(df, rr, predictors)
    predictions.to_csv('predictions.csv', index=False)


    # print(predictions)
    df = pd.read_csv('predictions.csv')

    plt.plot(predictions["actual"],predictions["prediction"])
    plt.xlabel('actual')
    plt.ylabel('prediction')
    plt.title('Graph of 3 columns from data frame')
    plt.show()
