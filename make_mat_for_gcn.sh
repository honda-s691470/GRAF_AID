#!/bin/bash

if [ "$#" -lt 4 ]; then
    echo "Usage: $0 <gwas> <snp_list> <bfile> <pheno_file> [k] [r2] [p_val] [population] [disease]"
    exit 1
fi

gwas=$1
snp_list=$2
bfile=$3
pheno_file=$4
k="${5:-5}"
p_val="${6:-0.01}"
r2="${7:-2}"
population="${8:-TEST}"
disease="${9:-DIS1}"

mkdir SNP_select
make_snp_list() {
    echo "snp_list is created in the SNP_select directory because nan was specified as the argument for snp_list"
    c_r2=$(echo "0.1 * $r2" | bc)
    plink --bfile ${bfile} --clump ${gwas} --clump-r2 ${c_r2} --clump-kb 250 --clump-p1 ${p_val}  --out GCN_MAT_snp_list_r2_0${r2}
    awk 'NR > 1 {print $3}' GCN_MAT_snp_list_r2_0${r2}.clumped > SNP_select/GCN_MAT_snp_list_r2_0${r2}.snp
    snp_list="SNP_select/GCN_MAT_snp_list_r2_0${r2}.snp"
}

process_gwas_data() {
    local population=$1

    output_dir=${disease}_${population}_r2_${r2}
    mkdir ${output_dir}

    awk 'NR==1 {
        for (i=1; i<=NF; i++) {
            if ($i == "P" || $i == "p") {
                col_index=i;
                break;
            }
        }
    }
    NR>1 { print $1, $col_index }' ${gwas} | sort -k1,1 -u | uniq > GWAS_SNP_P

    sort SNP_select/${snp_list} | join -1 1 -2 1 - <(awk -v p=${p_val} '($2 <= p){print $1, $2}' GWAS_SNP_P) | sort -u > merge_01
    echo "finish process_gwas_data"
    rm GWAS_SNP_P
}

generate_matrices() {
    local population=$1

    awk '{print $1}' merge_01 > snp_id_P_01.snp_ext
    plink --bfile ${bfile} --extract snp_id_P_01.snp_ext --make-bed --out GCN_MAT
    plink --bfile GCN_MAT --out GCN_MAT --recode vcf
    grep chr GCN_MAT.vcf | sort -k 3 > nohead_mat
    grep CHR GCN_MAT.vcf > id_list
    datamash transpose -t' ' < merge_01 > snp_id_P.T
    cat id_list nohead_mat > GCN_MAT.mtx
    datamash transpose < GCN_MAT.mtx > GCN_MAT.mtx.T
    sed '/[CHROM|POS|QUAL]/d' GCN_MAT.mtx.T | tr '/' ';' | sed -e 's/0;0/0/g' -e 's/0;1/1/g' -e 's/1;1/2/g' -e 's/.;./0/g' -e 's/_[0-9]*//g' >  GCN_MAT.mtx.T.clean
    echo "finish generate_matrices"
    rm merge_01 snp_id_P_01.snp_ext id_list nohead_mat GCN_MAT.vcf    
}

set_header() {
    {
        echo -e 'ID\tpheno'
        echo -e 'P\t0'
    } | paste -d " " - snp_id_P.T | expand -t 1 > new_header_space
    echo "finish set_header"
}

finish_processing() {
    local population=$1

    sort GCN_MAT.mtx.T.clean | expand -t 1 > GCN_MAT.mtx.T.clean.sort

    local phenofile_suffix="${pheno_file}"
    sort $phenofile_suffix > id_sort

    join id_sort GCN_MAT.mtx.T.clean.sort | sed -e s/^M// > GCN_MAT.T.clean.pheno
    cat new_header_space GCN_MAT.T.clean.pheno > ${output_dir}/GCN_MAT_pheno_SNP_${population}

    mv GCN_MAT.mtx ${output_dir}/
    rm GCN_MAT.mtx.T GCN_MAT.mtx.T.clean GCN_MAT.mtx.T.clean.sort GCN_MAT.T.clean.pheno new_header_space GCN_MAT.bed GCN_MAT.bim GCN_MAT.fam 
}


perform_kfold_cv() {
    local k=${k}

    # Create a shuffled version of the pheno_file
    local shuffled_file=$(mktemp)
    shuf $pheno_file > $shuffled_file

    mkdir -p "${output_dir}/cv"

    # Get the total number of samples
    local num_samples=$(wc -l < "${shuffled_file}")

    # Calculate the size of each fold
    local fold_size=$(( num_samples / k ))
    local remainder=$(( num_samples % k ))

    # Extract header
    local header="ID pheno"

    # Split the shuffled pheno file into k folds
    for ((i=0; i<k; i++)); do
        local start=$(( i * fold_size + 1 ))
        local end=$(( start + fold_size - 1 ))

        # Distribute the remainder to the validation sets
        if (( remainder > 0 )); then
            end=$(( end + 1 ))
            remainder=$(( remainder - 1 ))
        fi

        # Validation set
        echo "$header" > "${output_dir}/cv/cross_val_${i}"
        awk "NR >= $start && NR <= $end" "$shuffled_file" >> "${output_dir}/cv/cross_val_${i}"

        # Training set
        echo "$header" > "${output_dir}/cv/cross_train_${i}"
        awk "NR < $start || NR > $end" "$shuffled_file" >> "${output_dir}/cv/cross_train_${i}"
    done

    rm $shuffled_file
}

for popu in ${population}; do
    if [ "$snp_list" = "nan" ] || [ "$snp_list" = "Nan" ]; then
        make_snp_list
    fi
    echo $snp_list
    process_gwas_data $popu
    generate_matrices $popu
    set_header
    finish_processing $popu
    if [ "$k" -ne 0 ]; then
        perform_kfold_cv
    fi
done

