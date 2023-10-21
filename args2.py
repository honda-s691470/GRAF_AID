import argparse
import datetime


d_today = datetime.date.today()

parser = argparse.ArgumentParser()

#data 
parser.add_argument('--name', default='no_name',
                    help='name of analysis')  
parser.add_argument('--data_folder', default='DIS_TEST_r2_10',
                    help='data_folder')    
parser.add_argument('--train_val_data', default='GCN_MAT_pheno_SNP_TEST',
                    help='train_val_data')
parser.add_argument('--data_folder_test', default='DIS_TEST2_r2_10',
                    help='data_folder')
parser.add_argument('--test_data', default='GCN_MAT_pheno_SNP_TEST2',
                    help='test_data')
parser.add_argument('--GWAS_effect', default='None',
                    help='GWAS_effect')
parser.add_argument('--MHC', default='Include',
                    help='Include: use all snps, Exclude: exclude MHC resion (chr6:26000000-34000000), Only: use only MHC resion (chr6:26000000-34000000)',
                    choices=['Include', 'Exclude', 'Only'])
parser.add_argument('--Select_file', default='',
                    help='Mainly specifies files that have been LD clumped; files generated from PRSice or PLINK are accepted, but also files generated from make_mat_for_gcn.sh.')
parser.add_argument('--multi_task', default=[1],
                    help='Specify 1 as it is not yet implemented')
parser.add_argument('--fold_num', default=5, type=int,
                    help='number of fold')
parser.add_argument('--p_threshold', nargs='+', type=float, default=[0.01, 0.001, 1.0e-04, 1.0e-05, 1.0e-06, 1.0e-07, 1.0e-08])

#data for test
parser.add_argument('--multi_task_test', default=1, type=int,
                    help='Specify 1 as it is not yet implemented') 
parser.add_argument('--te_samp_max', default=10000, type=int)
parser.add_argument('--test_model_num', default=1, type=int,
                    help='model name to calculate feature importance')
parser.add_argument('--analysis_date_for_test', default = str(d_today),
                    help='date of analysis for test mode')

# scheduler
parser.add_argument('--lr', '--learning_rate', default=1e-3, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--scheduler', default='CosineAnnealingLR',
                    choices=['CosineAnnealingLR', 'ReduceLROnPlateau', 'MultiStepLR', 'ConstantLR'])
parser.add_argument('--min_lr', default=1e-6, type=float,
                    help='minimum learning rate')
parser.add_argument('--factor', default=0.5, type=float)
parser.add_argument('--patience', default=20, type=int)
parser.add_argument('--milestones', default='1,2', type=str)
parser.add_argument('--gamma', default=2/3, type=float)

#hyper_parameter
parser.add_argument('--epochs', default=400, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--num_test_model', default=2, type=int,
                    help='Number of GCNs you want to build out of the models registered in models.py (__all__). Constructed in order from the top.')
parser.add_argument('--num_class', default=2, type=int,
                    help='num_class')
parser.add_argument('--adj_metric', default='cosine',
                    help='adj_metric')
parser.add_argument('--adj_parameter', nargs='+', type=int, default=[2,3],
                    help='adj_parameter')
parser.add_argument('--init_weight', default='xn')
parser.add_argument('--reg', default = "elastic", type=str, choices=['elastic', 'l1', 'l2'])
parser.add_argument('--reg_alpha', default = 0.0001, type=float)
parser.add_argument('--l1l2_ratio', default = 0.99, type=float)
parser.add_argument('--dropout', default=0.5)

config = vars(parser.parse_args())
