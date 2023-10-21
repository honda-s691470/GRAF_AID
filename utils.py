import os
import re
import numpy as np
import datetime
import torch
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.optim import lr_scheduler
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, classification_report
from auc_delong_xu import auc_ci_Delong#https://github.com/RaulSanchezVazquez/roc_curve_with_confidence_intervals
from torchmetrics.functional import pairwise_manhattan_distance
import scipy.stats
from models import GraphConvolution
import torch.nn as nn
import models


cuda = True if torch.cuda.is_available() else False

def lrp(R, module, adj, input_tensor, linear_processed=False, trait=True):

    if isinstance(module, nn.Sequential):
        for submodule in module.children():
            if isinstance(submodule, nn.Linear) and linear_processed:
                continue
            R, linear_processed = lrp(R, submodule, adj, input_tensor, linear_processed)
            if isinstance(submodule, nn.Linear):
                linear_processed = True
        return R, linear_processed

    elif isinstance(module, GraphConvolution):
        input_tensor = input_tensor[0]
        z = torch.mm(input_tensor, module.weight)
        z_bias = z + module.bias.unsqueeze(0)  # adding bias
        z = torch.sparse.mm(adj, z_bias)  # use z_bias instead of z
        s = (R / z).sum(dim=1, keepdim=True)
        c = (input_tensor * s)
        return c, linear_processed

    elif isinstance(module, nn.Linear) and not linear_processed:
        input_tensor = input_tensor[0]
        z = torch.mm(input_tensor, module.weight.T)
        z_bias = z + module.bias.unsqueeze(0)  # adding bias
        if trait==True:
            z_sub = z_bias[:,1]  # use z_bias instead of z
        else:
            z_sub = z_bias[:,0]
        s = R / z_sub
        s = s.unsqueeze(1)
        c = (input_tensor * s)
        linear_processed = True
        return c, linear_processed

    else:
        raise ValueError("module type not covered in LRP")

def ext_te_samp (config, p, k, mode='test'):
    if mode == "val":
        id_tr = pd.read_csv(config["data_folder"] + "/" + f"p_thre_{p}" + "/"+ f'{k}'+ "_tr_val" + "/id_tr.csv")
        id_te = pd.read_csv(config["data_folder"] + "/" + f"p_thre_{p}" + "/"+ f'{k}'+ "_tr_val" + "/id_te.csv")
        id_trte = pd.concat([id_tr, id_te],axis=0)
        id_trte = list(id_trte.iloc[0:,0])
    
    elif mode == "test":
        lab_tr = np.loadtxt(os.path.join(config["data_folder_test"], f'p_thre_{p}', f'{0}_tr_val', f"labels_tr1.csv"))
        lab_te = np.loadtxt(os.path.join(config["data_folder_test"], f'p_thre_{p}', f'{0}_tr_val', f"labels_te1.csv"))
        lab_all = np.concatenate([lab_tr, lab_te], 0)
        cont = np.where(lab_all==0)
        case = np.where(lab_all==1)
        use_index = np.concatenate([case[0],cont[0][0:config['te_samp_max']-len(case[0])]],0)

        id_tr = pd.read_csv(config["data_folder_test"] + "/" + f"p_thre_{p}" + "/"+ f'{0}'+ "_tr_val" + "/id_tr.csv")
        id_te = pd.read_csv(config["data_folder_test"] + "/" + f"p_thre_{p}" + "/"+ f'{0}'+ "_tr_val" + "/id_te.csv")            
        id_trte = pd.concat([id_tr, id_te],axis=0)
        id_trte = list(id_trte.iloc[use_index,0]) 
    return id_trte

def make_lab_weight(multi_task, labels_trte_list, trte_idx):
    labels_tr_tensor_list = []
    labels_te_tensor_list = []
    label_tr_list = []
    label_te_list = []
    sample_weight_tr = []
    sample_weight_te = []
    trait_list = []
    cc_ratio = []
    crit_list = []
    num_class = []
    for i in range (multi_task): 
        trait_judge = np.all((labels_trte_list[i]==1.0) | (labels_trte_list[i]==0.0))
        if trait_judge == True:
            print("binary trait")
            case = np.count_nonzero(labels_trte_list[i][trte_idx["tr"]] == 1.0)
            cont = np.count_nonzero(labels_trte_list[i][trte_idx["tr"]] == 0.0)
            print("number of case control", "case ", case, " control ", cont)
            cc_ratio.append(cont/case)
            labels_tr_tensor_list.append(torch.LongTensor(labels_trte_list[i][trte_idx["tr"]]))
            labels_te_tensor_list.append(torch.LongTensor(labels_trte_list[i][trte_idx["te"]]))
            trait_list.append(trait_judge)
            sample_weight_tr.append(torch.FloatTensor(cal_sample_weight(labels_trte_list[i][trte_idx["tr"]], 2, use_sample_weight=False)))
            sample_weight_te.append(torch.FloatTensor(cal_sample_weight(labels_trte_list[i][trte_idx["te"]], 2, use_sample_weight=False)))
            weights = torch.tensor([1.0, cc_ratio[i]]).cuda()
            crit_list.append(torch.nn.CrossEntropyLoss(reduction='none', weight=weights))
            num_class.append(2)
        else:
            print("quantitative trait")
            labels_tr_tensor_list.append(torch.FloatTensor(labels_trte_list[i][trte_idx["tr"]]))
            labels_te_tensor_list.append(torch.FloatTensor(labels_trte_list[i][trte_idx["te"]]))   
            trait_list.append(trait_judge)
            sample_weight_tr.append(torch.FloatTensor(cal_sample_weight(labels_trte_list[i][trte_idx["tr"]], 1, use_sample_weight=False)))
            sample_weight_te.append(torch.FloatTensor(cal_sample_weight(labels_trte_list[i][trte_idx["te"]], 1, use_sample_weight=False)))
            crit_list.append(torch.nn.MSELoss(reduction='none'))  
            num_class.append(1)
        if cuda:
            label_tr_list.append(labels_tr_tensor_list[i].cuda())
            label_te_list.append(labels_te_tensor_list[i].cuda())
            sample_weight_tr[i] = sample_weight_tr[i].cuda()
            sample_weight_te[i] = sample_weight_te[i].cuda()
        torch.cuda.empty_cache()
    return label_tr_list, label_te_list, sample_weight_tr, sample_weight_te, trait_list, cc_ratio, crit_list, num_class

def make_gcn_list(config, data_tr, dim_list, num_class, h=None, gcn_list=None, mode='train_val'):
    if mode == "train_val":
        if config["name"] == 'SNN':
            gcn_name = [s for s in models.__all__ if 'SimpleNN' in s][h]
            gcn_list.append(models.__dict__[gcn_name](dim_list, num_class, config["dropout"], config["init_weight"]))
        elif config["MHC"] == 'Separate':
            gcn_name = models.__all__[h]
            gcn_list.append(models.__dict__[gcn_name](dim_list[0], num_class, config["dropout"], config["init_weight"]))
        else:
            gcn_name = models.__all__[h]
            gcn_list.append(models.__dict__[gcn_name](dim_list, num_class, config["dropout"], config["init_weight"]))
        print("gcn_name", gcn_name) 
        return gcn_list, gcn_name

    elif mode == "test":
        if config["name"] == 'SNN':
            print([s for s in models.__all__ if 'SimpleNN' in s])
            gcn_name = "SimpleNN_relu" #[s for s in models.__all__ if 'SimpleNN' in s][config["test_model_num"]-1]
            gcn_list = models.__dict__[gcn_name](dim_list, num_class, config["dropout"],config["init_weight"])
        elif config["MHC"] == 'Separate':
            gcn_name = models.__all__[config["test_model_num"]-1]
            gcn_list = models.__dict__[gcn_name](dim_list[0], num_class, config["dropout"],config["init_weight"])
        else:
            gcn_name = models.__all__[config["test_model_num"]-1] 
            gcn_list = models.__dict__[gcn_name](dim_list, num_class, config["dropout"],config["init_weight"])
        print("gcn_name", gcn_name) 
        return gcn_list, gcn_name

def make_result_mtx(config, multi_task, trait_list, auc_list, ci_delong_list, f1_list, acc_list, pcor_te_list, p, k, adj_para, gcn_name, model_num, dataframe):
    for l in range (multi_task):
        if trait_list[l] == True:
            print("Best AUC: {:.3f}".format(max(auc_list[l])))
            max_idx = np.array(auc_list[l]).argmax()
            print("95% CI", ci_delong_list[l][max_idx])
            print("Best F1: {:.3f}".format(f1_list[l][max_idx]))
            print("Best ACC: {:.3f}".format(acc_list[l][max_idx]))
        else:
            print(f"Best Corr{l}:" + "{:.3f}".format(max(pcor_te_list[l])))
        if True in trait_list:
            dataframe.append([p, k, l, adj_para, multi_task, gcn_name + f'_{multi_task}', config["dropout"], 
                              max(auc_list[l]), ci_delong_list[l][max_idx], acc_list[l][max_idx], f1_list[l][max_idx]])
        else:
            data_list = [p, k, l, adj_para, multi_task, gcn_name+ f'_{multi_task}', config["dropout"]]
            for l in range (multi_task):
                data_list = data_list + [max(pcor_te_list[l])]
            dataframe.append(data_list)
        #print("dataframe", dataframe)
    model_num += 1 
    return dataframe, model_num

def save_results(trait_list, dataframe, config, p, k, multi_task):
    if True in trait_list:
        stat_val = pd.DataFrame(dataframe).set_axis(['p', 'k', 'l', 'adj_para', 'multi_task', 'gcn_name', 'dropout', 'auc', '95%CI', 'acc', 'f1'], 
                                                    axis='columns', inplace=False)
        stat_val.to_csv(config["data_folder"] + "/" + f"p_thre_{p}" + "/" + f'{k}'+ "_tr_val" + "/result_" + 
                        config["analysis_date_for_test"] + "_MHC_" + config["MHC"] + config["Select_file"] + "_reg_" + config["reg"] + "_adj_" + str(config["adj_parameter"])+ ".csv", index=False)
    else:      
        col_list = ['p', 'k', 'l', 'adj_para', 'multi_task', 'gcn_name', 'dropout']
        for l in range (multi_task):
            col_list = col_list + [f'pcor{l+1}']
        stat_val = pd.DataFrame(dataframe).set_axis(col_list, axis='columns', inplace=False)
        stat_val['pcor_sum'] =0
        for l in range (multi_task):
            stat_val['pcor_sum'] = stat_val['pcor_sum'] + stat_val[f'pcor{l+1}']
        stat_val.to_csv(config["data_folder"] + "/" + f"p_thre_{p}" + "/" + f'{k}'+ "_tr_val" + "/result_" + 
                        config["analysis_date_for_test"] + "_MHC_" + config["MHC"] + config["Select_file"] + "_reg_" + config["reg"] + "_adj_" + str(config["adj_parameter"]) +".csv", index=False)
    torch.cuda.empty_cache()

def calc_lrp (ci, gcn, adj_trte, snp_list, k, config, mode='train_val', trait=True):
    if trait == True:
        R = ci[0][0:,1]
    else:
        R = ci[0][0:,0]
        #print("calc Trait", len(R), R)
    module_list = list(gcn.modules())  # Convert the generator to a list
    linear_processed = False
    for i, module in enumerate(reversed(module_list)):  # Now we can reverse the list
        if isinstance(module, (GraphConvolution, torch.nn.Linear)) and not isinstance(module, nn.Sequential):
            R, linear_processed = lrp(R, module, adj_trte, gcn.save_inputs[module].input_tensor, linear_processed, trait)
        elif isinstance(module, torch.nn.Sequential):
            for submodule in reversed(list(module.children())):  # The children() method also returns a generator, so we convert it to a list before reversing
                if isinstance(submodule, (GraphConvolution, torch.nn.Linear)) and not linear_processed:
                    R, linear_processed = lrp(R, submodule, adj_trte, gcn.save_inputs[submodule].input_tensor, linear_processed)

    #print(len(snp_list), R)
    # Create a DataFrame from the tensor R
    df = pd.DataFrame(R.cpu().detach().numpy(), columns=snp_list)

    # Save DataFrame to CSV
    if mode == "train_val":
        df.to_csv(f'{config["data_folder"]}/r_score{k}_{config["MHC"]}.csv', index=False) 
    elif mode == "test":
        df.to_csv(f'{config["data_folder_test"]}/r_score{k}_{config["MHC"]}.csv', index=False) 
   
def make_igraph(config, adj_trte, labels_trte_list, data_folder_trte, k):
    I = torch.eye(adj_trte.shape[0])
    adj_all_trte_I = adj_trte.to_dense().to('cpu').detach()-I
    os.makedirs("igraph_prac/" + config["data_folder"] + "/" + data_folder_trte + "/MHC_" + config["MHC"] + "_adj_" + str(config["adj_parameter"]) + "_p_thre_" + str(config["p_threshold"][0]) + "/" + str(k), exist_ok=True)
    pd.DataFrame(labels_trte_list).T.to_csv("igraph_prac/" + config["data_folder"] + "/" + data_folder_trte + "/MHC_" + config["MHC"] + "_adj_" + str(config["adj_parameter"]) + "_p_thre_" + str(config["p_threshold"][0]) + "/" + str(k) + f"/label_trte_cv_list_" + str(1) + "_full", index=False, header=None)
    # PyTorch tensor to numpy array
    adj_numpy = adj_all_trte_I.cpu().numpy()

    # Save numpy array
    np.save("igraph_prac/" + config["data_folder"] + "/" + data_folder_trte + "/MHC_" + config["MHC"] + "_adj_" + str(config["adj_parameter"]) + "_p_thre_" + str(config["p_threshold"][0]) + "/" + str(k) + '/adj_numpy.npy', adj_numpy)
 
def calc_reg_alpha(config, gcn_list, model_num):
    alpha = config["reg_alpha"]
    reg = torch.tensor(0., requires_grad=True)
    for w in gcn_list[model_num].parameters():
        if config["reg"]=="l1":
            reg = reg + torch.norm(w, 1)
        elif config["reg"]=="l2":
            reg = reg + torch.norm(w)**2
        elif config["reg"]=="elastic":
            reg = reg + (config["l1l2_ratio"])*torch.norm(w, 1) + (1-config["l1l2_ratio"])*torch.norm(w)**2
        else:
            reg = 0  
    return reg, alpha

def permutation_test_between_clfs(y_test, pred_proba_1, pred_proba_2, i, nsamples=10000):
    auc_differences = []
    pred_proba_1 = scipy.stats.zscore(pred_proba_1)
    pred_proba_2 = scipy.stats.zscore(pred_proba_2)
    auc1_ori = roc_auc_score(y_test.ravel(), pred_proba_1.ravel())
    auc2_ori = roc_auc_score(y_test.ravel(), pred_proba_2.ravel())
    observed_difference = auc1_ori - auc2_ori
    mask = np.random.randint(2, size=len(pred_proba_1.ravel()))
    for _ in range(nsamples):
        mask = np.random.randint(2, size=len(pred_proba_1.ravel()))
        p1 = np.where(mask, pred_proba_1.ravel(), pred_proba_2.ravel())
        p2 = np.where(mask, pred_proba_2.ravel(), pred_proba_1.ravel())
        auc1 = roc_auc_score(y_test.ravel(), p1)
        auc2 = roc_auc_score(y_test.ravel(), p2)
        auc_differences.append(auc1 - auc2)
    return config["p_threshold"][i], len(y_test.ravel()), list(y_test.ravel()).count(1), auc1_ori, auc2_ori, observed_difference, list((auc_differences >= observed_difference)).count(True)/nsamples

def cal_sample_weight(labels, num_class, use_sample_weight=True):
    if not use_sample_weight:
        return np.ones(len(labels)) / len(labels)
    count = np.zeros(num_class)
    for i in range(num_class):
        count[i] = np.sum(labels==i)
    sample_weight = np.zeros(labels.shape)
    for i in range(num_class):
        sample_weight[np.where(labels==i)[0]] = count[i]/np.sum(count)
    
    return sample_weight


def one_hot_tensor(y, num_dim):
    y_onehot = torch.zeros(y.shape[0], num_dim)
    y_onehot.scatter_(1, y.view(-1,1), 1)
    
    return y_onehot


def cosine_distance_torch(x1, x2=None, eps=1e-8):
    x2 = x1 if x2 is None else x2
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
    return 1 - torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)


def to_sparse(x):
    x_typename = torch.typename(x).split('.')[-1]
    sparse_tensortype = getattr(torch.sparse, x_typename)
    indices = torch.nonzero(x)
    if len(indices.shape) == 0:  # if all elements are zeros
        return sparse_tensortype(*x.shape)
    indices = indices.t()
    values = x[tuple(indices[i] for i in range(indices.shape[0]))]
    return sparse_tensortype(indices, values, x.size())


def cal_adj_mat_parameter(edge_per_node, data, metric="cosine"):
    #assert metric == "cosine", "Only cosine distance implemented"
    if metric == "cosine":
        dist = cosine_distance_torch(data, data)
    elif metric == "cosine_weight":
        dist = cosine_distance_torch(data, data)
        dist -= dist.min()
        dist /= dist.max()
        dist = dist**6
    print("dist.shape",dist.shape, edge_per_node, data.shape)
    parameter = torch.sort(dist.reshape(-1,)).values[edge_per_node*data.shape[0]]
    print(parameter)
    #return np.asscalar(parameter.data.cpu().numpy())
    return parameter.data.cpu().item()

def graph_from_dist_tensor(dist, parameter, self_dist=True):
    if self_dist:
        assert dist.shape[0]==dist.shape[1], "Input is not pairwise dist matrix"
    g = (dist <= parameter).float()
    if self_dist:
        diag_idx = np.diag_indices(g.shape[0])
        g[diag_idx[0], diag_idx[1]] = 0
        
    return g


def gen_adj_mat_tensor(data, parameter, metric):        
    if metric == "cosine":
        dist = cosine_distance_torch(data, data)
        g = graph_from_dist_tensor(dist, parameter, self_dist=True)
        adj = 1-dist
        adj = adj*g 
        adj_T = adj.transpose(0,1)        
    elif metric =="cosine_weight":
        adj = cosine_distance_torch(data, data)
        adj -= adj.min()
        adj /= adj.max()
        adj = adj**6
        g = graph_from_dist_tensor(adj, parameter, self_dist=True)
        adj = 1-adj
        adj = adj*g 
        adj_T = adj.transpose(0,1) 
    else:
        raise NotImplementedError
        
    adj = adj + adj_T*(adj_T > adj).float() - adj*(adj_T > adj).float()
    
    I = torch.eye(adj.shape[0])
    if cuda:
        I = I.cuda(0)
        
    if metric == "cosine":
        adj = F.normalize(adj + I, p=1)
    else:
        adj = adj + I
    adj_sparse = to_sparse(adj)
    
    return adj_sparse


def gen_test_adj_mat_tensor(data, trte_idx, parameter, metric):
    adj = torch.zeros((data.shape[0], data.shape[0]))
    if cuda:
        adj = adj.cuda()
    num_tr = len(trte_idx["tr"])
    
    if metric == "cosine":
        dist_tr2tr = cosine_distance_torch(data[trte_idx["tr"]], data[trte_idx["tr"]])
        g_tr2tr = graph_from_dist_tensor(dist_tr2tr, parameter, self_dist=False)
        adj[:num_tr,:num_tr] = 1-dist_tr2tr
        adj[:num_tr,:num_tr] = adj[:num_tr,:num_tr]*g_tr2tr
    elif metric =="cosine_weight":
        dist_tr2tr = cosine_distance_torch(data[trte_idx["tr"]], data[trte_idx["tr"]])
        dist_tr2tr -= dist_tr2tr.min()
        dist_tr2tr /= dist_tr2tr.max()
        dist_tr2tr = dist_tr2tr**6
        g_tr2tr = graph_from_dist_tensor(dist_tr2tr, parameter, self_dist=False)
        dist_tr2tr = 1-dist_tr2tr
        adj[:num_tr,:num_tr] = dist_tr2tr*g_tr2tr
    else:
        raise NotImplementedError
    
    #################################

    if metric == "cosine":
        dist_tr2te = cosine_distance_torch(data[trte_idx["tr"]], data[trte_idx["te"]])
        g_tr2te = graph_from_dist_tensor(dist_tr2te, parameter, self_dist=False)
        adj[:num_tr,num_tr:] = 1-dist_tr2te
        adj[:num_tr,num_tr:] = adj[:num_tr,num_tr:]*g_tr2te
    elif metric =="cosine_weight":
        dist_tr2te = cosine_distance_torch(data[trte_idx["tr"]], data[trte_idx["te"]])
        dist_tr2te = dist_tr2te - dist_tr2tr.min()
        dist_tr2te = dist_tr2te / dist_tr2tr.max()
        dist_tr2te = dist_tr2te**6
        g_tr2te = graph_from_dist_tensor(dist_tr2te, parameter, self_dist=False)
        adj[:num_tr,num_tr:] = 1-dist_tr2te
        adj[:num_tr,num_tr:] = adj[:num_tr,num_tr:]*g_tr2te
    else:
        raise NotImplementedError
 
    dist_te2tr = cosine_distance_torch(data[trte_idx["te"]], data[trte_idx["tr"]])
    
    if metric == "cosine":
        dist_te2tr = cosine_distance_torch(data[trte_idx["te"]], data[trte_idx["tr"]])
        g_te2tr = graph_from_dist_tensor(dist_te2tr, parameter, self_dist=False)
        adj[num_tr:,:num_tr] = 1-dist_te2tr
        adj[num_tr:,:num_tr] = adj[num_tr:,:num_tr]*g_te2tr # retain selected edges
    elif metric =="cosine_weight":
        dist_te2tr = cosine_distance_torch(data[trte_idx["te"]], data[trte_idx["tr"]])
        dist_te2tr = dist_te2tr - dist_tr2tr.min()
        dist_te2tr = dist_te2tr / dist_tr2tr.max()
        dist_te2tr = dist_te2tr**6
        g_te2tr = graph_from_dist_tensor(dist_te2tr, parameter, self_dist=False)
        adj[num_tr:,:num_tr] = 1-dist_te2tr
        adj[num_tr:,:num_tr] = adj[num_tr:,:num_tr]*g_te2tr
    else:
        raise NotImplementedError
    
    adj_T = adj.transpose(0,1)
    I = torch.eye(adj.shape[0])
    torch.cuda.empty_cache()
    if cuda:
        I = I.cuda()
    adj = adj + adj_T*(adj_T > adj).float() - adj*(adj_T > adj).float()
    if metric == "cosine":
        adj = F.normalize(adj + I, p=1)
    else:
        adj = adj + I
    adj = to_sparse(adj)
    
    return adj


def save_model_dict(folder, model_dict):
    if not os.path.exists(folder):
        os.makedirs(folder)
    for module in model_dict:
        torch.save(model_dict[module].state_dict(), os.path.join(folder, module+".pth"))
            
    
def load_model_dict(folder, model_dict):
    for module in model_dict:
        if os.path.exists(os.path.join(folder, module+".pth")):
#            print("Module {:} loaded!".format(module))
            model_dict[module].load_state_dict(torch.load(os.path.join(folder, module+".pth"), map_location="cuda:{:}".format(torch.cuda.current_device())))
        else:
            print("WARNING: Module {:} from model_dict is not loaded!".format(module))
        if cuda:
            model_dict[module].cuda()    
    return model_dict

def scheduler_maker(optimizer, config):
    if config['scheduler'] == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs']/1, eta_min=config['min_lr'])
    elif config['scheduler'] == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=config['factor'], patience=config['patience'], verbose=1, min_lr=config['min_lr'])
    elif config['scheduler'] == 'ConstantLR':
        scheduler = None
    else:
        raise NotImplementedError
    return scheduler

def performance_ind (labels_trte, trte_idx, prob):
    fpr, tpr, thresholds = roc_curve(labels_trte[trte_idx["te"]], prob[:,1])
    
    Youden_index_candidates = tpr-fpr
    index = np.where(Youden_index_candidates==max(Youden_index_candidates))[0][0]
    cutoff = thresholds[index]
    cutoff_result = np.where(prob[:,1] < cutoff, 0, 1)
   
    auc_temp = roc_auc_score(labels_trte[trte_idx["te"]], prob[:,1])
    auc_temp, _, auc_int = auc_ci_Delong(y_true=labels_trte[trte_idx["te"]], y_scores=prob[:,1])
    f1_temp = f1_score(labels_trte[trte_idx["te"]],cutoff_result)
    acc_temp = accuracy_score(labels_trte[trte_idx["te"]], cutoff_result) 
    cm = confusion_matrix(labels_trte[trte_idx["te"]], cutoff_result)
    
    return auc_temp, f1_temp, acc_temp, cm, cutoff_result, auc_int

def read_file_with_pattern(folder, pattern_str, i):
    file_list = os.listdir(folder)
    pattern = re.compile(pattern_str.format(i=i))
    matched_files = [f for f in file_list if pattern.match(f)]
    
    if matched_files:
        return pd.read_csv(os.path.join(folder, matched_files[0]), sep=" ", engine='python', dtype=str)
    else:
        print("No matching file found!")
        return None

def split_matrix (config):
    df = pd.read_csv(os.path.join(config["data_folder"], config["train_val_data"]), sep=" ", engine='python')
    for p in config["p_threshold"]:
        print("p_threshold", p)
        col = pd.DataFrame(df.iloc[0,1:].astype(float)<=p)
        col_name = ["ID"] + col[col[0]==True].index.to_list()
        print("number of snp",len(col_name))
        df_tr_val = df[col_name].iloc[1:,0:]
    
        folder_pattern = os.path.join(config["data_folder"], "cv")
        file_pattern_train = 'cross_train_.*{i}'
        file_pattern_val = 'cross_val_.*{i}'

        for i in range (config["fold_num"]):
            os.makedirs(config["data_folder"] + "/" + f"p_thre_{p}" + "/" + f'{i}'+ "_tr_val" + "/" + "weight", exist_ok=True) 
        
        feature = pd.DataFrame(df_tr_val.columns.tolist()[2:])
        for i in range(config["fold_num"]):
            feature.to_csv(config["data_folder"] + "/" + f"p_thre_{p}" + "/" + f'{i}'+ "_tr_val" + "/1_featname.csv", index=False, header=False)

        df_train_list=[]
        df_val_list=[]
        for i in range(config["fold_num"]):
            df_train = read_file_with_pattern(folder_pattern, file_pattern_train, i)
            df_val = read_file_with_pattern(folder_pattern, file_pattern_val, i)
            print("number of training sample in ", i, "fold ", len(df_train))
            print("number of validation sample in ", i, "fold ", len(df_val))
            if df_train is not None:
                df_train = df_train[["ID"]].astype(str)
                df_train_list.append(pd.merge(df_train, df_tr_val, on='ID'))
            if df_val is not None:
                df_val = df_val[["ID"]].astype(str)
                df_val_list.append(pd.merge(df_val, df_tr_val, on='ID'))

        for i in range(config["fold_num"]):
            id_tr = df_train_list[i]['ID']
            id_val = df_val_list[i]['ID']

            id_tr.to_csv(config["data_folder"] + "/" + f"p_thre_{p}" + "/"+ f'{i}'+ "_tr_val" + "/id_tr.csv", index=False, header=True)
            id_val.to_csv(config["data_folder"] + "/" + f"p_thre_{p}" + "/"+ f'{i}'+ "_tr_val" + "/id_te.csv", index=False, header=True)
                        
            train = df_train_list[i].iloc[0:, 2:].dropna(how='all', axis=1)
            val = df_val_list[i].iloc[0:, 2:].dropna(how='all', axis=1)

            train.to_csv(config["data_folder"] + "/" + f"p_thre_{p}" + "/"+ f'{i}'+ "_tr_val" + "/1_tr.csv", index=False, header=False)
            val.to_csv(config["data_folder"] + "/" + f"p_thre_{p}" + "/"+ f'{i}'+ "_tr_val" + "/1_te.csv", index=False, header=False)

            labels_tr = df_train_list[i]["pheno"]
            labels_val = df_val_list[i]["pheno"]
            labels_tr.to_csv(config["data_folder"] + "/" + f"p_thre_{p}" + "/"+ f'{i}'+ "_tr_val" + "/labels_tr1.csv", index=False, header=False)
            labels_val.to_csv(config["data_folder"] + "/"+ f"p_thre_{p}" + "/" + f'{i}'+ "_tr_val" + "/labels_te1.csv", index=False, header=False)           

def test_matrix (config):
    df = pd.read_csv(os.path.join(config["data_folder_test"], config["test_data"]), sep=" ", engine='python')
    print(df)
    for p in config["p_threshold"]:

        col = pd.DataFrame(df.iloc[0,1:].astype(float)<=p)
        col_name = ["ID"] + col[col[0]==True].index.to_list()
        print("number of snp",len(col_name))
        df_test = df[col_name].iloc[1:,0:]

        os.makedirs(config["data_folder_test"] + "/" + f"p_thre_{p}" + "/" + f'{0}'+ "_test", exist_ok=True)

        feature = pd.DataFrame(df_test.columns.tolist()[2:])
        feature.to_csv(config["data_folder_test"] + "/" + f"p_thre_{p}" + "/" + f'{0}'+ "_test" + "/1_featname.csv", index=False, header=False)

        id_test = df_test['ID']
        id_test.to_csv(config["data_folder_test"] + "/" + f"p_thre_{p}" + "/"+ f'{0}'+ "_test" + "/id_te.csv", index=False, header=True)
        test = df_test.iloc[0:, 2:].dropna(how='all', axis=1)
        test.to_csv(config["data_folder_test"] + "/" + f"p_thre_{p}" + "/"+ f'{0}'+ "_test" + "/1_te.csv", index=False, header=False)
        labels_test = df_test["pheno"]
        labels_test.to_csv(config["data_folder_test"] + "/"+ f"p_thre_{p}" + "/" + f'{0}'+ "_test" + "/labels_te1.csv", index=False, header=False)

def build_filename(config, p, fold_idx=None, function='read', prefix='result', ext='.csv'):
    base_str = f"{config['data_folder']}"
    if function == "read":
        base_str += f"/p_thre_{p}/{fold_idx}_tr_val"
        return f"{base_str}/{prefix}_{config['analysis_date_for_test']}_MHC_{config['MHC']}{config['Select_file']}_reg_{config['reg']}_adj_{config['adj_parameter']}{ext}"
    elif function == "output1":
        base_str += f"/p_thre_{p}/"
        return f"{base_str}/{prefix}_{config['analysis_date_for_test']}_multi_{config['multi_task']}_MHC_{config['MHC']}{config['Select_file']}_reg_{config['reg']}_adj_{config['adj_parameter']}{ext}"
    elif function == "output2":
        return f"{base_str}/{prefix}_{config['analysis_date_for_test']}_multi_{config['multi_task']}_MHC_{config['MHC']}{config['Select_file']}_reg_{config['reg']}_adj_{config['adj_parameter']}{ext}"

def make_statsum(config, trait_list):
    rank_list = []
    
    for p in config["p_threshold"]:
        dfs = []
        for i in range(config["fold_num"]):
            df = pd.read_csv(build_filename(config, p, i))
            
            column = "auc" if True in trait_list else "pcor1"
            df = df[["gcn_name", "dropout", "l", "adj_para", column]]
            
            dfs.append(df)

        merged_df = dfs[0]
        for df in dfs[1:]:
            merged_df = pd.merge(merged_df, df, on=["gcn_name", "dropout", "l", "adj_para"])

        df1 = merged_df.T.iloc[4:]
        df1.columns = merged_df["gcn_name"] + "_d" + merged_df["dropout"].astype(str) + "_l" + merged_df["l"].astype(str) + "_adj" + merged_df["adj_para"].astype(str)

        df1.astype(float).to_csv(build_filename(config, p, function='output1', prefix='result_sum'))
        
        df_stat = df1.astype(float).describe()
        df_stat.iloc[0] = p
        df_stat.to_csv(build_filename(config, p, function='output1', prefix='result_stat'))
        df_stat.to_csv(build_filename(config, p, function='output2', prefix='result_stat'), mode='a')
        
        rank_list.append(df_stat.iloc[0:2])

    rank_df = pd.concat([df.T for df in rank_list])
    rank_df.to_csv(build_filename(config, p, function='output2', prefix='rank_list'))


