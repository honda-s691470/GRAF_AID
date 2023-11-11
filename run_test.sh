#!/bin/bash
#SBATCH -p G
#SBATCH -J sq
#SBATCH -o stdout.%J
#SBATCH -e stderr.%J


if [ "$#" -lt 6 ]; then
    echo "Usage: $0 [data_folder] [data_folder_test] [test_data] [p_threshold] [fold_num] [MHC] [test_model_num] [Select_file] [analysis_date] [adj_parameter] [te_samp_max]"
    exit 1
fi

data_folder="${1:-DIS1_TEST_r2_2}"
data_folder_test="${2:-DIS1_TEST2_r2_2}"
test_data="${3:-GCN_MAT_pheno_SNP_TEST2}"
p_threshold="${4}"
fold_num="${5:-5}"
MHC="${6:-Only}"
test_model_num="${7:-3}"
Select_file="${8}" #:-GCN_MAT_snp_list_r2_02.snp}"
analysis_date="${9:-$(date +%m-%d-%Y)}"
adj_parameter="${10:-3}"
te_samp_max="${11:-10000}"

IFS=' ' read -ra p_threshold_array <<< "$p_threshold"
for p in "${p_threshold_array[@]}"; do
    p_threshold_args+=" $p"
done

if [ -n "$Select_file" ]; then
    Select_file_arg="--Select_file ${Select_file}"
else
    Select_file_arg=""
fi

python test_matrix.py --data_folder_test ${data_folder_test} --test_data ${test_data} --p_threshold ${p_threshold_args}
python run_test.py --data_folder ${data_folder} --data_folder_test ${data_folder_test} --p_threshold ${p_threshold_args} --MHC ${MHC} ${Select_file_arg} --analysis_date_for_test ${analysis_date} --adj_parameter ${adj_parameter} --test_model_num ${test_model_num} --te_samp_max ${te_samp_max} --fold_num ${fold_num}


#nohup sh run_test_UC_QC.sh > output_test_UC_fromWB.log 2>&1 &

