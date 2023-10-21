# GRAF_AID

GRAF-AID, short for Graph-based Risk Assessment For Auto Immune Disease, is a tool designed to predict the onset of autoimmune diseases. Utilizing Graph Convolutional Networks (GCN), this model addresses the complexities of the Human Leukocyte Antigen (HLA) region.

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

