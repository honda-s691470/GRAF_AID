# GRAF_AID

GRAF-AID, short for Graph-based Risk Assessment For Auto Immune Disease, is a tool designed to predict the onset of autoimmune diseases. Utilizing Graph Convolutional Networks (GCN), this model addresses the complexities of the Human Leukocyte Antigen (HLA) region.

# Data preparation

bash make_mat_for_gcn.sh PATH_TO_GWASsumstat PATH_TO_snp_list PATH_TO_plink_bfile PATH_TO_phenotype_file number_of_fold p_value population disease_name r2_value  

- PATH_TO_GWASsumstat
  
