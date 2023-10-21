#!/bin/bash

if [ "$#" -lt 6 ]; then
    echo "Usage: $0 [data_folder] [train_val_data] [p_threshold] [fold_num] [MHC] [num_test_model] [Select_file] [analysis_date]"
    exit 1
fi

data_folder="${1:-DIS1_TEST_r2_2}"
train_val_data="${2:-GCN_MAT_pheno_SNP_TEST}"
p_threshold="${3}"
fold_num="${4:-5}"
MHC="${5:-Only}"
num_test_model="${6:-3}"
Select_file="${7:-GCN_MAT_snp_list_r2_02.snp}"
analysis_date="${8:-$(date +%m-%d-%Y)}"

IFS=' ' read -ra p_threshold_array <<< "$p_threshold"
for p in "${p_threshold_array[@]}"; do
    p_threshold_args+=" $p"
done

python split.py --data_folder ${data_folder} --train_val_data ${train_val_data} --p_threshold ${p_threshold_args} --fold_num ${fold_num}
python run_tr_val.py --data_folder ${data_folder} --p_threshold ${p_threshold_args} --fold_num ${fold_num} --MHC ${MHC} --Select_file ${Select_file} --num_test_model ${num_test_model} --analysis_date_for_test ${analysis_date}
