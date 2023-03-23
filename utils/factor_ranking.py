import pandas as pd
import numpy as np
import time
from multiprocessing import Process
from Nan_operation import del_invalid_column,del_nan_row
from rank_calculation import calculate_process_per_company
from pickle_operation import pickle_store

'''
this file is to do the feature selection based on the research paper
'''



if __name__=="__main__":

    training_factors=pd.read_csv('../data/train_factor_processed.csv').set_index(['date','code'])
    training_factors=training_factors.unstack()

    training_return=pd.read_csv('../data/first_round_train_return_data.csv')
    training_return['code']=training_return['date_time'].apply(lambda x:x[0:x.index('d')])
    training_return['date']=training_return['date_time'].apply(lambda x:x[x.index('d')+1:])
    training_return['date']=training_return['date'].apply(int)
    training_return.drop(['date_time'],axis=1,inplace=True)
    training_return=training_return.set_index(['date','code'])
    training_return=training_return.unstack()

    training_data=pd.merge(training_factors,training_return,on=['date'],how='left')
    training_data=training_data.stack()



    #Remove columns with null values greater than 0.1 due to rolling
    training_data=del_invalid_column(training_data)

    #Remove data older than 60 days
    training_data=del_nan_row(training_data)

    training_data=training_data.stack(level=0).unstack()
    pickle_store(training_data,'../data/train_factor_with_return.pckl')



    ranking_dict={}
    print("现在开始因子排序，针对个股，每隔三十天计算一次排序")

    T1=time.time()
    company_list=list(set(training_data.columns.get_level_values(0)))
    for index in range(len(company_list)):
        if index%6==0:
            P0=Process(target=calculate_process_per_company,args=(training_data,company_list[index]))
            P1=Process(target=calculate_process_per_company,args=(training_data,company_list[index+1]))
            P2=Process(target=calculate_process_per_company,args=(training_data,company_list[index+2]))
            P3=Process(target=calculate_process_per_company,args=(training_data,company_list[index+3]))
            P4=Process(target=calculate_process_per_company,args=(training_data,company_list[index+4]))
            P5=Process(target=calculate_process_per_company,args=(training_data,company_list[index+5]))
            #进程启动
            P0.start()
            P1.start()
            P2.start()
            P3.start()
            P4.start()
            P5.start()
            #主进程挂起
            P0.join()
            P1.join()
            P2.join()
            P3.join()
            P4.join()
            P5.join()




    T2=time.time()
    print("因子排序历时",T2-T1)










