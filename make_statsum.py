import pandas as pd
import datetime
import models
from args2 import config

rank_list = []
d_today = config["analysis_date_for_test"]
for p in config["p_threshold"]:
    dir_name = config["data_folder"] + "/p_thre_" + str(p)
    for i in range(config["fold_num"]):
        if i==0:
            df0 = pd.read_csv(dir_name + "/" + f'{i}' + "_tr_val" + "/result_" + str(d_today) +   "_MHC_" + config["MHC"] + config["Select_file"] + "_reg_" + config["reg"]+"_adj_" + str(config["adj_parameter"]) + ".csv")
            if True in trait_list:
                df0 = df0[["gcn_name", "dropout", "l", "adj_para", "auc"]]
            else:
                df0 = df0[["gcn_name", "dropout", "l", "adj_para", "pcor1"]]
        else:
            df = pd.read_csv(dir_name + "/" + f'{i}' + "_tr_val" + "/result_" + str(d_today) +   "_MHC_" + config["MHC"] + config["Select_file"] + "_reg_" + config["reg"]+ "_adj_" + str(config["adj_parameter"]) + ".csv")
            if True in trait_list:
                df = df[["gcn_name", "dropout", "l", "adj_para", "auc"]]      
            else:
                df = df[["gcn_name", "dropout", "l", "adj_para", "pcor1"]] 
            df0 = pd.merge(df0, df, on=["gcn_name", "dropout", "l", "adj_para"])
    print(df0.T)
    print(df0["gcn_name"] + "_d" + df0["dropout"].astype(str))
    df1 = df0.T.iloc[4:,0:]
    print(df1.astype(float).describe())
    df1.columns = df0["gcn_name"]+  "_d" + df0["dropout"].astype(str)+ "_l" + df0["l"].astype(str)+"_adj" + df0["adj_para"].astype(str)  
    df1.astype(float).to_csv(dir_name + "/" +"result_sum_" + str(d_today) +  "_multi_" + str(config["multi_task"]) + "_MHC_" + config["MHC"] + config["Select_file"] + "_reg_" + config["reg"] + "_adj_" + str(config["adj_parameter"]) + ".csv")
    df_stat = df1.astype(float).describe()
    df_stat.iloc[0,0:] = p 
    df_stat.to_csv(dir_name + "/" +"result_stat_" + str(d_today) + ".csv")
    df_stat.to_csv(config["data_folder"] + "/" +"result_stat_" + str(d_today) + "_multi_" + str(config["multi_task"]) + "_MHC_" + config["MHC"] + config["Select_file"] + "_reg_" + config["reg"]+ "_adj_" + str(config["adj_parameter"]) +".csv", mode='a')
    rank_list.append(df_stat.iloc[0:2,0:])

rank_list[0] = rank_list[0].T
for p in range(len(config["p_threshold"])-1):
    rank_list[0] = pd.concat([rank_list[0],rank_list[p+1].T],axis=0)
    
rank_list[0].to_csv(config["data_folder"] + "/" +"rank_list" + str(d_today) + "_multi_" + str(config["multi_task"]) + "_MHC_" + config["MHC"] + config["Select_file"] + "_reg_" + config["reg"]+ "_adj_" + str(config["adj_parameter"]) +".csv")
