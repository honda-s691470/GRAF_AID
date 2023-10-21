#!/bin/bash
#SBATCH -p G
#SBATCH -J sq
#SBATCH -o stdout.%J
#SBATCH -e stderr.%J


if [ "$#" -lt 6 ]; then
    echo "Usage: $0 [data_folder] [train_val_data] [data_folder_test] [test_data] [p_threshold] [fold_num] [MHC] [test_model_num] [Select_file] [analysis_date] [adj_parameter] [te_samp_max]"
    exit 1
fi

data_folder="${1:-DIS1_TEST_r2_2}"
train_val_data="${2:-GCN_MAT_pheno_SNP_TEST}"
data_folder_test="${3:-DIS1_TEST2_r2_2}"
test_data="${4:-GCN_MAT_pheno_SNP_TEST2}"
p_threshold="${5}"
fold_num="${6:-5}"
MHC="${7:-Only}"
test_model_num="${8:-3}"
Select_file="${9:-GCN_MAT_snp_list_r2_02.snp}"
analysis_date="${10:-$(date +%m-%d-%Y)}"
adj_parameter="${11:-3}"
te_samp_max="${12:-10000}"

IFS=' ' read -ra p_threshold_array <<< "$p_threshold"
for p in "${p_threshold_array[@]}"; do
    p_threshold_args+=" $p"
done

python test_matrix.py --data_folder_test ${data_folder_test} --test_data ${test_data} --p_threshold ${p_threshold_args}
python run_test.py --data_folder ${data_folder} --data_folder_test ${data_folder_test} --p_threshold ${p_threshold_args} --MHC ${MHC} --Select_file ${Select_file} --analysis_date_for_test ${analysis_date} --adj_parameter ${adj_parameter} --test_model_num ${test_model_num} --te_samp_max ${te_samp_max} --fold_num ${fold_num}


#nohup sh run_test_UC_QC.sh > output_test_UC_fromWB.log 2>&1 &

