import math
from pickle_operation import pickle_store
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
'''
this is a function that calculate the rank of all the factors we contribute 
under the current time period
'''


def calculate(df:pd.DataFrame,company,training_start_day,validation_start_day,prediction_start_day):

    training_data=df[company].loc[training_start_day:validation_start_day-1,:]
    validation_data=df[company].loc[validation_start_day:prediction_start_day-1,:]

    training_data_y=training_data.pop('return')
    training_data_x=training_data

    validation_data_y=validation_data.pop('return')
    validation_data_x=validation_data




    ranking_list=[]
    current_max_value_list=[]
    go_down_time=0
    for i in range(training_data_x.shape[1]):
        current_value_dict = {}
        for j in range(i,training_data_x.shape[1]):

            #If the factor pool does not yet have a factor,
            # then we do a univariate regression to find the factor with the largest marginal effect
            if len(ranking_list)==0:
                regressor=LinearRegression()
                regressor.fit(np.array(training_data_x.iloc[:, j]).reshape(-1, 1), training_data_y)
                value = regressor.score(np.array(validation_data_x.iloc[:, j]).reshape(-1, 1), validation_data_y)
                current_value_dict[training_data_x.columns[j]] = value

            #If the factor pool has factors, we bring in the factors from the factor pool
            else:
                T = math.ceil(10 * math.log(len(ranking_list) + 1))
                D = math.ceil(2 * math.log(len(ranking_list) + 1))
                regressor = RandomForestRegressor(n_estimators=T, max_depth=D,n_jobs=8)
                #Add a factor to the previous ranking and calculate the score after adding this factor
                current_train_x=training_data_x.loc[:,ranking_list]
                current_train_x.insert(0,training_data_x.columns[j],training_data_x.iloc[:,j])

                current_validation_x = validation_data_x.loc[:, ranking_list]
                current_validation_x.insert(0, validation_data_x.columns[j], validation_data_x.iloc[:, j])

                regressor.fit(current_train_x, training_data_y)
                value = regressor.score(current_validation_x, validation_data_y)
                current_value_dict[training_data_x.columns[j]] = value

        current_factor_with_max_value=max(current_value_dict,key=current_value_dict.get)

        if len(ranking_list) <= 5:
            ranking_list.append(current_factor_with_max_value)
            training_data_x.insert(0,current_factor_with_max_value,training_data_x.pop(current_factor_with_max_value))
            validation_data_x.insert(0,current_factor_with_max_value,validation_data_x.pop(current_factor_with_max_value))
            current_max_value_list.append(max(current_value_dict.values()))
        else:
            #After having at least five factors, we first check whether adding the variables makes the score higher
            if max(current_value_dict.values())<=current_max_value_list[-1]:
                go_down_time+=1
            else:
                go_down_time=0

            ranking_list.append(current_factor_with_max_value)

            training_data_x.insert(0, current_factor_with_max_value,
                                   training_data_x.pop(current_factor_with_max_value))
            validation_data_x.insert(0, current_factor_with_max_value,
                                     validation_data_x.pop(current_factor_with_max_value))
            current_max_value_list.append(max(current_value_dict.values()))

            if go_down_time==3:
                #If it appears that all three variables added in succession drop in score,
                # then we stop adding variables and find the set of factors with the highest score
                # at that point
                break

    ranking_list=ranking_list[:list.index(current_max_value_list,max(current_max_value_list))+1]

    print(company,"在时期",training_start_day,"的因子排序为",ranking_list)
    print(company, "在时期", training_start_day, "的因子列表长度为", len(ranking_list))
    return ranking_list


def calculate_process_per_company(training_data:pd.DataFrame,company):
    print("正在计算股票", company, "的因子排序", "*" * 15)
    ranking_dict=dict()
    training_start_day = 61
    # For each stock, we update the factor ranking every 30 days
    validation_start_day = training_start_day + 181
    prediction_start_day = validation_start_day + 31
    while prediction_start_day <= 1000:
        # If the training and validation sets can be fetched in their entirety,
        # then we do a factorial sort
        try:
            current_ranklist = calculate(training_data, company, training_start_day, validation_start_day,
                                         prediction_start_day)
            training_start_day += 31
            validation_start_day = training_start_day + 181
            prediction_start_day = validation_start_day + 31
            ranking_dict[company + str(training_start_day)] = current_ranklist

        except:
            training_start_day += 31
            validation_start_day = training_start_day + 181
            prediction_start_day = validation_start_day + 31
            print('时期', training_start_day - 31, "因子排序出错")
            ranking_dict[company + str(training_start_day)] = []





    pickle_store(ranking_dict,'../data/'+str(company)+'_factor_ranking.pckl')














