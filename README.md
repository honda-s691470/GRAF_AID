# GRAF_AID

GRAF-AID, short for Graph-based Risk Assessment For Auto Immune Disease, is a tool designed to predict the onset of autoimmune diseases. Utilizing Graph Convolutional Networks (GCN), this model addresses the complexities of the Human Leukocyte Antigen (HLA) region.

<img src="https://github.com/honda-s691470/GRAF_AID/assets/80377824/3d70c9c8-59d3-4069-bcef-646b9ab1cae2" width="512">

# Data preparation
Create a file named GCN_MAT_pheno_SNP_${population} in the directory ${disease}_${population}_r2_${r2} using the following code.
```sh
bash make_mat_for_gcn.sh PATH_TO_GWAS_sumstat PATH_TO_snp_list PATH_TO_plink_bfile PATH_TO_phenotype_file number_of_fold p_value population disease_name r2_value  
```
make_mat_for_gcn.sh uses datamash; if you do not have datamash installed, install it with the following code.
```
conda install -c bioconda datamash
```
The first four arguments (PATH_TO_GWAS_sumstat, PATH_TO_snp_list, PATH_TO_plink_bfile, PATH_TO_phenotype_file) must be entered.

- **PATH_TO_GWAS_sumstat:** Full path to the GWAS sumstat file. The first column is the SNP ID of type chr:position and somewhere in the column must contain a GWAS p-value with column name p/P.

```
SNP CHR BP A1 A2 P BETA
chr1:1118275 1 1118275 T C 0.017228570369999998 0.1570892152427751
chr1:1120431 1 1120431 A G 0.6123953074 -0.020815139713920003
chr1:1135242 1 1135242 C A 0.9469519648 0.003992021269537457
chr1:1140435 1 1140435 T G 0.8636767822 0.007174203748000453
...
```
- **PATH_TO_snp_list:** Full path to the snp_list file.
```
SNP
chr1:1118275 
chr1:1135242
...
```
To align PRSice-2 with SNP, you can use the --print-snp option in PRSice-2 to create a *.snp file, then use e.g.
```sh
awk '{print $2}' *.snp > *.snp_ext
```
If there is no SNP file, the second argument nan/Nan will create an SNP file in the SNP_select directory from the specified gwas file, bfile, p_value and r2_value. Note that this creation method does not use LD reference.
```sh
bash make_mat_for_gcn.sh PATH_TO_GWAS_sumstat non PATH_TO_plink_bfile PATH_TO_phenotype_file number_of_fold 0.01 population disease_name 2  
```
- **PATH_TO_plink_bfile:** Full path to the plink bfile. No extension required. The second column of the bim file must be of type chr:pos.
```
1  chr1:1118275  0  1118275 T C
1  chr1:1120431  0  1120431 A G
1  chr1:1135242  0  1135242 C A
1  chr1:1140435  0  1140435 T G
...
```
- **PATH_TO_phenotype_file:** Full path to the phenotype file. The first column of the phenotype file must be an ID and the second column must be a binary type with 1 as case and 0 as control, or a continuous value.
```
001 1
002 1
003 0
004 0
...
```
- **number_of_fold:** Specifies an integer for creating a file for k-fold cross validation. If 0 is specified, no file is created. The default value is 5.
- **p_value:** Specifies the maximum p-value of SNPs in the created file; it is recommended that a value between 0.01 and 0.05 be specified as large p-values result in very large file sizes. The default value is 0.01.
- **population:** Used as part of the name of the directory to be created. The default is TEST.
- **disease_name:** Used as part of the directory part of the directory to be created. Defaults to DIS1.
- **r2_value:** Used as part of the name of the directory to be created. If Nan/nan is specified in PATH_TO_snp_list, the value specified in this r2_value is entered into plink to create the snp_list. The default value is 2.

# Training & Validation
Training & Validtaion uses the run_tr_val.sh file. The basic arguments are: data_folder train_val_data p_threshold fold_num MHC num_test_model Select_file analysis_date. When specifically using the arguments The following.
```sh
sh run_tr_val.sh DIS1_TEST_r2_2 GCN_MAT_pheno_SNP_TEST "1e-07 1e-08" 3 Only 3 GCN_MAT_snp_list_r2_02.snp "10-17-2023"
```
- **data_folder:** Specify the name of the data folder; if created with the default values of make_mat_for_gcn.sh, the folder name will be DIS1_TEST_r2_2.
- **train_val_data:** Specify the name of the train_val_data - if it was created with the default value of make_mat_for_gcn.sh, the data GCN_MAT_pheno_SNP_TEST is created in the folder DIS1_TEST_r2_2, so "GCN_MAT_pheno_SNP_TEST" is the argument.
- **p_threshold:** Multiple p-values can be specified. To specify multiple p-values, enclose the values in "". Example: "1e-07 1e-08".
- **fold_num:**　k Specifies the value of k for fold cross validation, the same value as the number_of_fold argument in make_mat_for_gcn.sh. Specifically, the sample ID file created in the cv directory in the data_folder is read.
- **MHC:** Choose from Include, Exclude and Only. Include: use all snps, Exclude: exclude MHC resion (chr6:26000000-34000000 /hg19), Only: use only MHC resion (chr6:26000000-34000000 /hg19)
- **num_test_model:** Number of GCNs you want to build out of the models registered in models.py (__all__). Constructed in order from the top.
- **Select_file:** Mainly specifies files that have been LD clumped; files generated from PRSice or PLINK are accepted, but also files generated from make_mat_for_gcn.sh. If you created the file GCN_MAT_snp_list_r2_02.snp in the SNP_select directory using the default values in make_mat_for_gcn.sh, the argument should be "GCN_MAT_snp_list_r2_02.snp". The argument should be "GCN_MAT_snp_list_r2_02.snp".
- **analysis_date:** Specifies the date of the analysis. If not stated, today's date is automatically taken as the argument.

# Test
Test uses the run_test.sh file. The basic arguments are: data_folder data_folder_test test_data p_threshold fold_num MHC test_model_num Select_file analysis_date adj_parameter te_samp_max. When specifically using the arguments The following.
```sh
sh run_test.sh DIS1_TEST_r2_2 DIS1_TEST2_r2_2 GCN_MAT_pheno_SNP_TEST2 "1e-08" 3 Only 2 GCN_MAT_snp_list_r2_02.snp "10-17-2023" 3 10000
```
- **data_folder:** Specify the name of the data folder; if created with the default values of make_mat_for_gcn.sh, the folder name will be DIS1_TEST_r2_2.
- **data_folder_test:** Specify the name of the data folder test. It can be created using make_mat_for_gcn.sh, but the SNP must be the same as the file created in Trainig & Validation.
- **test_data:** Specify the name of the test_data.
- **p_threshold:** Specify one of the p-values you want in Trainig & Validation Check the rank_list_*.csv created in the data_folder of Trainig & Validation (by default DIS1_TEST_r2_2) to see which model you want to test. Check which models are to be tested. Example: "1e-08".
- **fold_num:**　k Specifies the value of k for fold cross validation, the same value as the number_of_fold argument in make_mat_for_gcn.sh. Specifically, the sample ID file created in the cv directory in the data_folder is read.
- **MHC:** Select the arguments used in Training & Validation from Include, Exclude and Only. Include: use all snps, Exclude: exclude MHC resion (chr6:26000000-34000000 /hg19), Only: use only MHC resion (chr6:26000000-34000000 /hg19)
- **test_model_num:** Enter the number of the model to be tested. The models and numbers correspond to the following. 1:GCN_E2_decline_L2_log_clf1_selu_multi, 2:GCN_E2_decline_L2_div10_clf1_selu_multi, 3:GCN_E2_decline_L2_sqr_clf1_selu_multi, 4:GCN_E2_decline_L3_sqr_clf1_selu_multi, 5:GCN_E2_decline_L4_sqr_clf1_selu_multi, 6:GCN_E2_decline_L5_sqr_clf1_selu_multi, 7:SimpleNN_relu
- **Select_file:** Mainly specifies files that have been LD clumped; files generated from PRSice or PLINK are accepted, but also files generated from make_mat_for_gcn.sh. Specify the same file as used in Training & Validation.
- **analysis_date:** Specifies the date of the analysis. If not stated, today's date is automatically taken as the argument. Specify the date when the Training & Validation was carried out.
- **adj_parameter:** Check the rank_list_*.csv created in the Trainig & Validation data_folder (default: DIS1_TEST_r2_2) for the adj_parameter of the model to be tested. Example: 2.
- **te_samp_max:** Graph convolutional networks require exponentially more GPU memory as the number of samples or SNPs increases. If GPU memory is insufficient, this argument can be used to reduce the number of samples. In this case, the samples are randomly sampled from data_folder_test.
