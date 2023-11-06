""" Training and testing of the model
"""
import os
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import torch
import torch.nn.functional as F
import seaborn as sns
from matplotlib import pyplot as plt
import models
from  models import get_model_with_hooks
from utils import *
import re
import glob
import torch_optimizer as optim
from scipy.stats import pearsonr
import math
from torchmetrics import PearsonCorrCoef
import itertools
import statistics
#from torch_geometric.data import Data
#from torch_geometric.datasets import Planetoid  
#from torch_geometric.explain import Explainer, GNNExplainer  
#from torch_geometric.nn import GCNConv  


cuda = True if torch.cuda.is_available() else False


def load_csv(config, p, k, fname, **kwargs):
    path = os.path.join(config["data_folder"], f'p_thre_{p}', f'{k}_tr_val', fname)
    return pd.read_csv(path, **kwargs)

def process_mhc(config, data_tr, data_te, snp_list, snp_list_merge):
    if config["MHC"] in ["Include", "Exclude", "Only"]:
        snp_list_df = snp_list_merge.iloc[0:,0].str.split(':', expand=True)
        snp_list_df.columns = ["chr", "pos"]
        snp_list_df["chr:pos"] = snp_list_merge
        snp_list_df["pos"] = snp_list_df["pos"].astype(int)
    
        query_str = {
            "Include": 'chr:pos',
            "Exclude": 'not (chr == "chr6" & pos > 26000000 & pos < 34000000)',
            "Only": '(chr == "chr6" & pos > 26000000 & pos < 34000000)'
        }.get(config["MHC"], '')
        snp_filtered_df = snp_list_df.query(query_str)

    elif config["MHC"] in ["Impute"]:
        snp_list_df = snp_list.iloc[0:,0].str.split('_', expand=True)
        snp_list_df.columns = ["chr", "pos"]
        snp_list_df["chr:pos"] = snp_list
        snp_list_df["pos"] = snp_list_df["pos"].astype(int)
        snp_filtered_df = snp_list_df

    ext_index = snp_list[snp_list["chr:pos"].isin(snp_filtered_df["chr:pos"])].index.tolist()

    print("MHC flag is ", config["MHC"] , ", number of SNPs used for analysis", len(ext_index))

    return data_tr[:,ext_index], data_te[:,ext_index], snp_filtered_df["chr:pos"]

def prepare_trte_data_multi(config, p, k, multi_task, mode="train_val"):
    np.random.seed(42)
    snp_list = load_csv(config, p, k, "1_featname.csv", sep=",", header=None)
    samp_id = None
    print("number of SNPs including 1_featname.csv", len(snp_list))
    if config["Select_file"]:
        snp_list.columns = ["chr:pos"]
        snp_list_select = pd.read_csv("SNP_select/" + config["Select_file"], header=None)
        snp_list_select.columns = ["chr:pos"]
        print("number of SNPs including SNP Select file",len(snp_list_select))
        snp_list_merge = pd.merge(snp_list, snp_list_select, on="chr:pos")
        print("number of SNPs after merge", len(snp_list_merge))
    else:
        snp_list.columns = ["chr:pos"]
        snp_list_merge = snp_list

    if mode == "train_val":
        
        labels_tr_list = [np.loadtxt(os.path.join(config["data_folder"], f'p_thre_{p}', f'{k}_tr_val', f"labels_tr{i+1}.csv")) for i in range(multi_task)]
        labels_te_list = [np.loadtxt(os.path.join(config["data_folder"], f'p_thre_{p}', f'{k}_tr_val', f"labels_te{i+1}.csv")) for i in range(multi_task)]

        id_tr = pd.read_csv(config["data_folder"] + "/" + f"p_thre_{p}" + "/"+ f'{k}'+ "_tr_val" + "/id_tr.csv", header=None, skiprows=1)
        id_te = pd.read_csv(config["data_folder"] + "/" + f"p_thre_{p}" + "/"+ f'{k}'+ "_tr_val" + "/id_te.csv", header=None, skiprows=1)
        id_te.columns = id_tr.columns
        print("id_te.columns", id_te.columns)
        samp_id = pd.concat([id_tr, id_te],axis=0)
        samp_id = list(samp_id.iloc[0:,0])

        data_tr = np.loadtxt(os.path.join(config["data_folder"], f'p_thre_{p}', f'{k}_tr_val', "1_tr.csv"), delimiter=',')
        data_te = np.loadtxt(os.path.join(config["data_folder"], f'p_thre_{p}', f'{k}_tr_val', "1_te.csv"), delimiter=',')
 
        if config["MHC"] in ["Include", "Exclude", "Only", "Impute"]:
            data_tr, data_te, snp_list = process_mhc(config, data_tr, data_te, snp_list, snp_list_merge)

        if config["GWAS_effect"] != "None":
            eff_list = pd.read_csv(config["GWAS_effect"], sep=" ", header=None, engine='python')
            snp_eff_list = pd.merge(snp_list, eff_list, on=0)
            effect = np.array(snp_eff_list.iloc[0:,1]).reshape(1,-1)
            data_tr *= effect
            data_te *= effect

    elif mode == "test":
        labels_tr_list = [np.loadtxt(os.path.join(config["data_folder"], f'p_thre_{p}', f'{k}_tr_val', f"labels_tr{i+1}.csv")) for i in range(multi_task)]
        labels_all = [np.loadtxt(os.path.join(config["data_folder_test"], f'p_thre_{p}', f'{0}_test', f"labels_te{i+1}.csv")) for i in range(multi_task)]

        id_tr = pd.read_csv(config["data_folder"] + "/" + f"p_thre_{p}" + "/"+ f'{k}'+ "_tr_val" + "/id_tr.csv", header=None, skiprows=1)
        id_te = pd.read_csv(config["data_folder_test"] + "/" + f"p_thre_{p}" + "/"+ f'{0}'+ "_test" + "/id_te.csv", header=None, skiprows=1)

        min_length = min([len(label) for label in labels_all])
        num_samples = min(min_length, config['te_samp_max'])

        use_index = np.random.choice(min_length, num_samples, replace=False)
        labels_te_list = [label[use_index] for label in labels_all]
        id_te = id_te.iloc[use_index,0]
        id_te.columns = id_tr.columns
        samp_id = pd.concat([id_tr, id_te],axis=0)
        samp_id = list(samp_id.iloc[0:,0])
        data_tr = np.loadtxt(os.path.join(config["data_folder"], f'p_thre_{p}', f'{k}_tr_val', f"1_tr.csv"), delimiter=',')
        data_te = np.loadtxt(os.path.join(config["data_folder_test"], f'p_thre_{p}', f'{0}_test', f"1_te.csv"), delimiter=',')
        data_te = data_te[use_index]
        
        if config["MHC"] in ["Include", "Exclude", "Only", "Impute"]:
            data_tr, data_te, snp_list = process_mhc(config, data_tr, data_te, snp_list, snp_list_merge)

        if config["GWAS_effect"] != "None":
            eff_list = pd.read_csv(config["GWAS_effect"], sep=" ", header=None, engine='python')
            snp_eff_list = pd.merge(snp_list, eff_list, on=0)
            effect = np.array(snp_eff_list.iloc[0:,1]).reshape(1,-1)
            data_tr *= effect
            data_te *= effect

    num_tr = data_tr.shape[0]
    num_te = data_te.shape[0]
    print("number of training data ", num_tr, ", number of validation/test data ", num_te)

    label_list = []
    if config["MHC"] in ["Include", "Exclude", "Only", "Impute"]:
        data_mat = np.concatenate((data_tr, data_te), axis=0)
        data_tensor = torch.FloatTensor(data_mat)
        data_tensor = data_tensor.cuda()

        idx_dict = {}
        idx_dict["tr"] = list(range(num_tr))
        idx_dict["te"] = list(range(num_tr, (num_tr+num_te)))

        data_train = data_tensor[idx_dict["tr"]].clone()
        data_test = data_tensor[idx_dict["te"]].clone()
        data_all = torch.cat((data_tensor[idx_dict["tr"]].clone(),data_tensor[idx_dict["te"]].clone()),0)
        for i in range (multi_task):
            label_list.append(np.concatenate((labels_tr_list[i], labels_te_list[i])))
    return data_train, data_test, data_all, idx_dict, label_list, snp_list, samp_id
        

def gen_trte_adj_mat(data_tr, data_trte, trte_idx, adj_para, config):
    if config["adj_metric"] == "cosine":
        adj_parameter_adaptive = cal_adj_mat_parameter(adj_para, data_tr, config["adj_metric"])
    elif config["adj_metric"] == "cosine_weight":
        adj_parameter_adaptive = cal_adj_mat_parameter(adj_para, data_tr, config["adj_metric"])
    elif config["adj_metric"] == "manh":
        adj_parameter_adaptive = cal_adj_mat_parameter(adj_para, data_tr, config["adj_metric"])
    adj_train = gen_adj_mat_tensor(data_tr, adj_parameter_adaptive, config["adj_metric"])
    adj_test = gen_test_adj_mat_tensor(data_trte, trte_idx, adj_parameter_adaptive, config["adj_metric"])
    
    return adj_parameter_adaptive, adj_train, adj_test

def gen_trte_adj_mat_test(data_tr, data_trte, trte_idx, adj_parameter_adaptive, config):
    adj_train = gen_adj_mat_tensor(data_tr, adj_parameter_adaptive, config["adj_metric"])
    adj_test = gen_test_adj_mat_tensor(data_trte, trte_idx, adj_parameter_adaptive, config["adj_metric"])
    
    return adj_train, adj_test

class TrainingContext:
    def __init__(self, config, gcn_list, model_num, optimizer, data_tr, adj_tr, trte_idx, multi_task, 
                 trait_list, crit_list, label_tr_list, sample_weight_tr, scheduler, dim_list, train_loss_list):
        self.config = config
        self.gcn_list = gcn_list
        self.model_num = model_num
        self.optimizer = optimizer
        self.data_tr = data_tr
        self.adj_tr = adj_tr
        self.trte_idx = trte_idx
        self.multi_task = multi_task
        self.trait_list = trait_list
        self.crit_list = crit_list
        self.label_tr_list = label_tr_list
        self.sample_weight_tr = sample_weight_tr
        self.pcor_tr_list = [[] for _ in range(5)]
        self.scheduler = scheduler
        self.dim_list = dim_list
        self.train_loss_list = train_loss_list
        self.pearson = PearsonCorrCoef().cuda()

    def _handle_trait(self, l, ci):
        prob = F.softmax(ci[l], dim=1).data.cpu().numpy()
        ci_loss_tr = torch.mean(torch.mul(self.crit_list[l](ci[l], self.label_tr_list[l]), self.sample_weight_tr[l]))
        return [ci_loss_tr, 0]

    def _handle_non_trait(self, l, ci):
        prob = ci[l].data.cpu().numpy()
        ci_loss_tr = torch.mean(self.crit_list[l](ci[l], self.label_tr_list[l]))
        pcor = self.pearson(torch.squeeze(ci[l]), torch.squeeze(self.label_tr_list[l]))
        cor_loss = (1-pcor)/10
        pcor_tr_temp = pcor.data.cpu().numpy()
        self.pcor_tr_list[l].append(pcor_tr_temp)
        return [ci_loss_tr, cor_loss]

    def train(self):
        self.gcn_list[self.model_num].train()
        self.optimizer.zero_grad()
        ci = self.gcn_list[self.model_num](self.data_tr, self.adj_tr)
        reg, alpha = calc_reg_alpha(self.config, self.gcn_list, self.model_num)

        ci_loss_tr_vals = []
        cor_loss_vals = []
        tr_idx = self.trte_idx["tr"]

        for l in range(self.multi_task):
            ci[l] = ci[l][tr_idx, :]
            if self.trait_list[l]:
                ci_loss_tr_l, cor_loss_l = self._handle_trait(l, ci)
            else:
                ci_loss_tr_l, cor_loss_l = self._handle_non_trait(l, ci)
            ci_loss_tr_vals.append(ci_loss_tr_l)
            cor_loss_vals.append(cor_loss_l)

        ci_loss_tr = sum(ci_loss_tr_vals) + sum(cor_loss_vals)
        ci_loss_tr = ci_loss_tr + alpha*(self.dim_list/2.0e+5)*reg
        ci_loss_tr.backward()

        self.optimizer.step()
        self._handle_scheduler(ci_loss_tr)

        self.train_loss_list.append(ci_loss_tr.item())
        return self.train_loss_list, self.pcor_tr_list

    def _handle_scheduler(self, ci_loss_tr):
        if self.config["scheduler"] == 'CosineAnnealingLR':
            self.scheduler.step()
        elif self.config["scheduler"] == 'ReduceLROnPlateau':
            self.scheduler.step(ci_loss_tr)
        elif self.config["scheduler"] == 'ConstantLR':
            pass

class ValidationContext:
    def __init__(self, config, gcn_list, model_num, data_trte, adj_trte, trte_idx, multi_task, 
                 crit_list, label_te_list, sample_weight_te, labels_trte_list, trait_list, best_auc, best_pcor, p, k, h, adj_para):
        self.config = config
        self.gcn_list = gcn_list
        self.model_num = model_num
        self.data_trte = data_trte
        self.adj_trte = adj_trte
        self.trte_idx = trte_idx
        self.multi_task = multi_task
        self.crit_list = crit_list
        self.label_te_list = label_te_list
        self.sample_weight_te = sample_weight_te
        self.labels_trte_list = labels_trte_list
        self.trait_list = trait_list
        self.best_auc = best_auc
        self.best_pcor = best_pcor
        self.auc_list = [[] for _ in range(5)]
        self.f1_list = [[] for _ in range(5)]
        self.acc_list = [[] for _ in range(5)]
        self.cut_off_list = [[] for _ in range(5)]
        self.ci_delong_list = [[] for _ in range(5)]
        self.pcor_te_list = [[] for _ in range(5)]
        self.val_loss_list = []
        self.p = p
        self.k = k
        self.h = h
        self.adj_para = adj_para

    def _handle_trait(self, l, ci, auc_temp_list):
        try:
            ci[l] = ci[l][self.trte_idx["te"], :]
            prob = F.softmax(ci[l], dim=1).data.cpu().numpy()
            ci_loss_te = torch.mean(torch.mul(self.crit_list[l](ci[l], self.label_te_list[l]), self.sample_weight_te[l]))
            auc_temp, f1_temp, acc_temp, cm, cutoff_result, auc_int = performance_ind(self.labels_trte_list[l], self.trte_idx, prob)
            self.auc_list[l].append(auc_temp)
            auc_temp_list.append(auc_temp)
            self.f1_list[l].append(f1_temp)
            self.acc_list[l].append(acc_temp)
            self.cut_off_list[l].append(cutoff_result)
            self.ci_delong_list[l].append(auc_int)
            return [ci_loss_te, 0]
        except ValueError as e:
            print(e)
            for lst in [self.auc_list[l], self.f1_list[l], self.acc_list[l], self.cut_off_list[l], self.ci_delong_list[l]]:
                lst.append(0)
            return [0, 0]

    def _handle_non_trait(self, l, ci):
        pearson = PearsonCorrCoef().cuda()
        ci[l] = ci[l][self.trte_idx["te"], :]
        prob = ci[l].data.cpu().numpy()
        ci_loss_te = torch.mean(self.crit_list[l](ci[l], self.label_te_list[l]))
        pcor = pearson(torch.squeeze(ci[l]), torch.squeeze(self.label_te_list[l]))
        cor_loss = (1-pcor)/10
        pcor_te_temp = pcor.data.cpu().numpy()
        self.pcor_te_list[l].append(pcor_te_temp)
        return [ci_loss_te, cor_loss]

    def validate(self):
        self.gcn_list[self.model_num].eval()
        with torch.no_grad():
            ci = self.gcn_list[self.model_num](self.data_trte, self.adj_trte)

            ci_loss_te = []
            cor_loss = []
            auc_temp_list = []

            for l in range(self.multi_task):
                if self.trait_list[l]:
                    ci_loss_te_l, cor_loss_l = self._handle_trait(l, ci, auc_temp_list)
                else:
                    ci_loss_te_l, cor_loss_l = self._handle_non_trait(l, ci)
                ci_loss_te.append(ci_loss_te_l)
                cor_loss.append(cor_loss_l)

            self.val_loss_list.append((sum(ci_loss_te) + sum(cor_loss)).item())

            if True in self.trait_list:
                try:
                    if np.mean(auc_temp_list) > self.best_auc:
                        self._save_model("auc")
                        self.best_auc = np.mean(auc_temp_list)
                except (PermissionError, RuntimeError) as e:
                    print(e)
            else:
                try:
                    if sum(self.pcor_te_list[l]) > self.best_pcor:
                        self._save_model("pcor")
                        self.best_pcor = sum(self.pcor_te_list[l])
                except PermissionError as e:
                    print(e)

    def _save_model(self, mode):
        path = f"{self.config['data_folder']}/p_thre_{self.p}/{self.k}_tr_val/weight/{self.config['analysis_date_for_test']}_weight_best_{mode}_model_num_%s_multi_%s_MHC_%s%s_reg_%s_adj_%s.pth"
        path = path % (self.h+1, self.multi_task, self.config["MHC"], self.config["Select_file"], self.config["reg"], str(self.adj_para))
        torch.save(self.gcn_list[self.model_num].state_dict(), path)

def train_val_multi(config):
    prob_cv_list = []
    for p in config["p_threshold"]:
        print("====== p threshold", p, " =====" )
        for k in range(config["fold_num"]): 
            print("====== ", k+1, " th fold train and validation =====" )  
            dataframe = []
            model_num = 0
            gcn_list=[]
            for multi_task in config["multi_task"]:   
                print("====== number of multitask", multi_task, " =====" )
                data_tr, data_te, data_trte, trte_idx, labels_trte_list, snp_list, samp_id = prepare_trte_data_multi(config, p, k, multi_task)
                dim_list=len(data_tr[0])
                
                label_tr_list, label_te_list, sample_weight_tr, sample_weight_te, trait_list, cc_ratio, crit_list, num_class = make_lab_weight(multi_task, labels_trte_list, trte_idx)

                for adj_para in config["adj_parameter"]:
                    print("====== adj parameter ", adj_para,  " =====")
                    adj_parameter_adaptive, adj_tr, adj_trte = gen_trte_adj_mat(data_tr, data_trte, trte_idx, adj_para, config)

                    for h in range(config["num_test_model"]):
                        gcn_list, gcn_name = make_gcn_list(config, data_tr, dim_list, num_class, h, gcn_list, mode='train_val')

                        gcn_list[model_num].cuda()                        
                        optimizer = torch.optim.Adam(gcn_list[model_num].parameters(), lr=config["lr"])
                        scheduler = scheduler_maker(optimizer, config)

                        #Start learning
                        train_loss_list = []
                        val_loss_list = []
                        best_auc, best_pcor = 0.10, 0.001

                        for l in range(multi_task):
                            if trait_list[l] == True:
                                pass
                            else:

                                tcat = torch.cat((label_tr_list[l], label_te_list[l]),dim=0)
                                label_tr_list[l] = torch.reshape((label_tr_list[l] - torch.min(tcat))/(torch.max(tcat)-torch.min(tcat)), (-1,1))
                                label_te_list[l] = torch.reshape((label_te_list[l] - torch.min(tcat))/(torch.max(tcat)-torch.min(tcat)), (-1,1))
                        train_context = TrainingContext(config, gcn_list, model_num, optimizer, data_tr, adj_tr, trte_idx, multi_task, trait_list, crit_list, label_tr_list, sample_weight_tr, scheduler, dim_list, train_loss_list)
                        val_context = ValidationContext(config, gcn_list, model_num, data_trte, adj_trte, trte_idx, multi_task, crit_list, label_te_list, sample_weight_te, labels_trte_list, trait_list, best_auc, best_pcor, p, k, h, adj_para) 

                        for epoch in range(1, config["epochs"]+1):
                            # Change network to learning mode
                            train_context.train() 
                            
                            # Change network to validation mode
                            val_context.validate()                     
 
                        # make sumstat matrix 
                        dataframe, model_num = make_result_mtx(config, multi_task, trait_list, val_context.auc_list, val_context.ci_delong_list, val_context.f1_list, val_context.acc_list, val_context.pcor_te_list, p, k, adj_para, gcn_name, model_num, dataframe)
 
            #save results 
            save_results(trait_list, dataframe, config, p, k, multi_task)
    make_statsum(config, trait_list)

def get_weight_id(config, p):
    result_list = glob.glob(os.path.join(config["data_folder"], f'p_thre_{p}', 'result_stat*.csv'))
    result_df = pd.read_csv(result_list[-1], sep=",")
    te_model_num = result_df.columns.to_list().index(config["test_model"])-1
    
    drop = float(config["test_model"].split('_')[-1].replace('d', ''))
    model_name = "_".join(config["test_model"].split('_')[0:-2])
    return te_model_num, drop, model_name

class ProbabilityCalculator:
    def __init__(self, config):
        self.config = config
        self.pearson = PearsonCorrCoef().cuda()
        self.prob = [[] for _ in range(5)]

    def handle_trait_task(self, l, ci, trte_idx, samp_id, p, k, trait_list, labels_trte_list):
        try:
            te_idx = trte_idx["te"]
            auc, f1, acc, _, cutoff_result, auc_ci = performance_ind(
                labels_trte_list[l], trte_idx, F.softmax(ci[l], dim=1).data.cpu().numpy()[te_idx, :])
            print("auc, 95%CI, f1, acc", auc, auc_ci, f1, acc)
            temp_df = pd.DataFrame(F.softmax(ci[l], dim=1).data.cpu().numpy()[:, 1])
            temp_df.index = samp_id
            temp_df.columns = [f'{p}_{k}']
            return temp_df, 0
        except ValueError as e:
            print(e)
            return None, 0

    def handle_non_trait_task(self, l, ci, label_te_list, trte_idx):
         
        pcor = self.pearson(torch.squeeze(ci[l][trte_idx["te"], :]), torch.squeeze(label_te_list[l]))
        print("pcor", pcor.item())
        return (1 - pcor) / 10

    def make_prob(self):
        for p in self.config["p_threshold"]:
            print(f"====== p threshold {p} =====")
            for k in range(self.config["fold_num"]):
                print(f"======test using best model in {k + 1} th fold======")
                print("best_model_num", self.config["test_model_num"])

                # loading and preprocessing steps
                data_tr, data_te, data_trte, trte_idx, labels_trte_list, snp_list, samp_id = prepare_trte_data_multi(self.config, p, k, self.config["multi_task_test"])
                dim_list=len(data_tr[0])
                label_tr_list, label_te_list, sample_weight_tr, sample_weight_te, trait_list, cc_ratio, crit_list, num_class = make_lab_weight(self.config["multi_task_test"], labels_trte_list, trte_idx)

                adj_parameter_adaptive, adj_tr, adj_trte = gen_trte_adj_mat(data_tr, data_trte, trte_idx, self.config["adj_parameter"], self.config)   

                gcn_list=[]
                gcn, gcn_name = make_gcn_list(self.config, data_tr, dim_list, num_class, None, None, mode='test')
                print(gcn_name) 
                gcn = models.__dict__[gcn_name]
                ModelWithHooks = get_model_with_hooks(gcn)
                gcn = ModelWithHooks(dim_list, num_class, self.config["dropout"], self.config["init_weight"])
                gcn.cuda()
                if True in trait_list:
                    mode = "auc"
                else:
                    mode= "pcor"
                weight_path = os.path.join(self.config["data_folder"], f'p_thre_{p}',f'{k}_tr_val','weight', f'%s_weight_best_{mode}_model_num_%s_multi_%s_MHC_%s%s_reg_%s_adj_%s.pth' % (self.config["analysis_date_for_test"], self.config["test_model_num"], self.config["multi_task_test"], self.config["MHC"], self.config["Select_file"], self.config["reg"], str(self.config["adj_parameter"])))
                print(weight_path)

                gcn.load_state_dict(torch.load(weight_path))
                gcn.eval()
    
                ci_list = []
                with torch.no_grad():
                    ci = gcn(data_trte, adj_trte)
                    for l in range(self.config["multi_task_test"]):
                        if trait_list[l]:
                            temp_df, _ = self.handle_trait_task(l, ci, trte_idx, samp_id, p, k, trait_list, labels_trte_list)
                            if temp_df is not None:
                                self.prob[l].append(temp_df)
                        else:
                            self.handle_non_trait_task(l, ci, label_te_list, trte_idx)

                # LRP and igraph steps
                if "SimpleNN" not in gcn_name:
                    calc_lrp(ci, gcn, adj_trte, snp_list, k, self.config, trait=trait_list[0])
                    make_igraph(self.config, adj_trte, labels_trte_list, self.config["data_folder"], k) 

        for l in range(self.config["multi_task_test"]):
            if trait_list[l]:
                for i in range(len(self.config["p_threshold"]) * self.config["fold_num"]):
                    if i != 0:
                        self.prob[l][0] = pd.merge(self.prob[l][0], self.prob[l][i], left_index=True, right_index=True, how='outer')
                # Save to CSV
                file_name = f'/train_val_prob_%s_%s_multi_%s_model_num_%s_MHC_%s%s_reg_%s_adj_%s.csv'
                path = self.config["data_folder"] + file_name % (
                    self.config["data_folder"], self.config["analysis_date_for_test"], str(l), self.config["test_model_num"],
                    self.config["MHC"], self.config["Select_file"], self.config["reg"], str(self.config["adj_parameter"]))
                self.prob[l][0].to_csv(path, index=True)

def make_prob(config):
    prob_calc = ProbabilityCalculator(config)
    prob_calc.make_prob()

class TestProcessor:
    def __init__(self, config):
        self.config = config
        self.prob_sum = [[] for _ in range(5)]
        self.trait_list = []
        self.prob_te = [[] for _ in range(5)]

    def _load_data(self, p, k):
        data_tr, data_te, data_trte, trte_idx, labels_trte_list, snp_list, samp_id = prepare_trte_data_multi(self.config, p, k, self.config["multi_task_test"], mode='test')
        return data_tr, data_te, data_trte, trte_idx, labels_trte_list, snp_list, samp_id

    def _initialize_variables(self):
        self.featname_list = []
        self.feat_imp_list = []
        self.perform_list = []
        self.cor_loss = []
        self.prob = [[] for _ in range(5)]

    def _calculate_probabilities(self, ci, labels_trte_list, label_te_list, trte_idx, trait_list, p, k):
        pearson = PearsonCorrCoef().cuda()
        for l in range(self.config["multi_task_test"]):
            try:
                if trait_list[l] == True:
                    self.prob[l].append(F.softmax(ci[l], dim=1).data.cpu().numpy())
                    self.prob_te[l].append(self.prob[l][k][trte_idx["te"],:])
                    self.cor_loss.append(0)
                    auc, f1, acc, _, cutoff_result, auc_ci = performance_ind(labels_trte_list[l], trte_idx, self.prob[l][k][trte_idx["te"],:])
                    self.perform_list.append([p, auc, auc_ci, f1, acc])
                    print(self.perform_list)
                else:
                    self.prob[l].append(ci[l].data.cpu().numpy())
                    self.prob_te[l].append(ci[l][trte_idx["te"],:].cpu().numpy())
                    pcor = pearson(torch.squeeze(ci[l][trte_idx["te"], :]), torch.squeeze(label_te_list[l]))
                    self.perform_list.append([p, pcor.item()])
                    print("pcor", pcor.item())
                    self.cor_loss.append((1 - pcor) / 10)
            except ValueError:
                self.perform_list.append([p, 0, 0, 0, 0])

    def test(self):
        for p in self.config["p_threshold"]:
            self._initialize_variables()

            for k in range(self.config["fold_num"]):
                data_tr, data_te, data_trte, trte_idx, labels_trte_list, snp_list, samp_id = self._load_data(p, k)
           
                label_tr_list, label_te_list, sample_weight_tr, sample_weight_te, trait_list, cc_ratio, crit_list, num_class = make_lab_weight(self.config["multi_task_test"], labels_trte_list, trte_idx)
                adj_parameter_adaptive, adj_tr, adj_trte = gen_trte_adj_mat(data_tr, data_trte, trte_idx, self.config["adj_parameter"], self.config)

                gcn_list = []
                gcn, gcn_name = make_gcn_list(self.config, data_tr, len(data_tr[0]), num_class, None, gcn_list, mode='test')
                print(gcn_name)
                gcn = models.__dict__[gcn_name]
                ModelWithHooks = get_model_with_hooks(gcn)
                gcn = ModelWithHooks(len(data_tr[0]), num_class, self.config["dropout"], self.config["init_weight"])
                gcn.cuda()
                if True in trait_list:
                    mode = "auc"
                else:
                    mode= "pcor"
                weight_path = os.path.join(self.config["data_folder"], f'p_thre_{p}',f'{k}_tr_val','weight', f'%s_weight_best_{mode}_model_num_%s_multi_%s_MHC_%s%s_reg_%s_adj_%s.pth' % (self.config["analysis_date_for_test"], self.config["test_model_num"], self.config["multi_task_test"], self.config["MHC"], self.config["Select_file"], self.config["reg"], str(self.config["adj_parameter"])))
                print(weight_path)

                gcn.load_state_dict(torch.load(weight_path))
                gcn.eval()

                with torch.no_grad():
                    ci = gcn(data_trte, adj_trte)
                    self._calculate_probabilities(ci, labels_trte_list, label_te_list, trte_idx, trait_list, p, k)

                if self.config["name"] != 'SNN':
                    calc_lrp(ci, gcn, adj_trte, snp_list, k, self.config, mode='test', trait=trait_list[0])
                    make_igraph(self.config, adj_trte, labels_trte_list, self.config["data_folder_test"], k)

                for l in range(self.config["multi_task_test"]):
                    self.prob_sum[l] = sum(self.prob_te[l]) / self.config["fold_num"]

            all_data = pd.DataFrame(self.perform_list)
            if trait_list[0] == True:
                all_data.columns = ["p_val", "auc", "95%CI", "f1", "acc"]
            else:
                all_data.columns = ["p_val", "pcor"]
            all_data_stat = all_data.describe()
            file_name = f'/test_result_stat_{self.config["data_folder"]}_{self.config["analysis_date_for_test"]}_multi_{self.config["multi_task_test"]}_model_num_{self.config["test_model_num"]}_MHC_{self.config["MHC"]}{self.config["Select_file"]}_reg_{self.config["reg"]}_adj_{str(self.config["adj_parameter"])}.csv'
            all_data_stat.to_csv(self.config["data_folder_test"] + file_name, index=True, mode='a')

        for l in range(self.config["multi_task_test"]):
            if trait_list[0] == True:
                prob_df = pd.DataFrame(self.prob_sum[l][0:, 1])
            else:
                prob_df = pd.DataFrame(self.prob_sum[l])
                
            columns = [f'{j}_{i}' for i in range(self.config["multi_task_test"]) for j in self.config["p_threshold"]]
            prob_df.columns = columns
            prob_df.index = [samp_id[i] for i in trte_idx["te"]]
            file_name = f'/test_prob_{self.config["data_folder"]}_{self.config["analysis_date_for_test"]}_multi_{self.config["multi_task_test"]}_model_num_{self.config["test_model_num"]}_MHC_{self.config["MHC"]}{self.config["Select_file"]}_reg_{self.config["reg"]}_adj_{str(self.config["adj_parameter"])}.csv'
            prob_df.to_csv(self.config["data_folder_test"] + file_name, index=True)

def test(config):
    test_proc = TestProcessor(config)
    test_proc.test()
            


    
