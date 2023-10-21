# GRAF_AID

GRAF-AID, short for Graph-based Risk Assessment For Auto Immune Disease, is a tool designed to predict the onset of autoimmune diseases. Utilizing Graph Convolutional Networks (GCN), this model addresses the complexities of the Human Leukocyte Antigen (HLA) region.

# Data preparation

bash make_mat_for_gcn.sh PATH_TO_GWASsumstat PATH_TO_snp_list PATH_TO_plink_bfile PATH_TO_phenotype_file number_of_fold p_value population disease_name r2_value  

- PATH_TO_GWASsumstat Full path to the directory that contains GWAS sumstat file. The first column is the SNP ID of type chr:position and somewhere in the column must contain a GWAS p-value with column name p/P.

```
SNP CHR BP A1 A2 P BETA
chr1:1118275 1 1118275 T C 0.017228570369999998 0.1570892152427751
chr1:1120431 1 1120431 A G 0.6123953074 -0.020815139713920003
chr1:1135242 1 1135242 C A 0.9469519648 0.003992021269537457
chr1:1140435 1 1140435 T G 0.8636767822 0.007174203748000453
,,,
```  
