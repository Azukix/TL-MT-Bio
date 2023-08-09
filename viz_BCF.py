
import numpy as np
from utils import build_dataset
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from utils.MY_GNN import collate_molgraphs, EarlyStopping, run_an_eval_predmerge,run_an_eval_epoch_heterogeneous_return_weight_py2,run_an_eval_epoch_pih,\
 set_random_seed, MGA
from utils.MY_GNN import pos_weight
import os
import time
import pandas as pd
start = time.time()

args = {}
args['device'] = "cuda" if torch.cuda.is_available() else "cpu"
args['atom_data_field'] = 'atom'
args['bond_data_field'] = 'etype'
args['classification_metric_name'] = 'roc_auc'
args['regression_metric_name'] = 'r2'
# model parameter
args['num_epochs'] = 300
args['patience'] = 50
args['batch_size'] = 256
args['mode'] = 'higher'
args['in_feats'] = 40
args['rgcn_hidden_feats'] = [512, 512]
args['classifier_hidden_feats'] = 512
args['rgcn_drop_out'] = 0.2
args['drop_out'] = 0.2
args['lr'] = 3
args['weight_decay'] = 5
args['loop'] = True

# task name (model name)
args['task_name'] = 'MT_BCFBAFBMF512512512' # change
args['data_name'] = 'KOWall4'  # change
args['times'] = 1

args['select_task_list'] = ['BCF','BAF','BMF']    # change
args['select_task_index'] = []
args['classification_num'] = 0
args['regression_num'] = 0
args['all_task_list'] = ['BCF','BAF','BMF']   # change
# generate select task index
for index, task in enumerate(args['all_task_list']):
    if task in args['select_task_list']:
        args['select_task_index'].append(index)
# generate classification_num
for task in args['select_task_list']:
    if task in []:
        args['classification_num'] = args['classification_num'] + 1
    if task in ['BCF','BAF','BMF']:
        args['regression_num'] = args['regression_num'] + 1

# generate regression_num
if args['classification_num'] != 0 and args['regression_num'] != 0:
    args['task_class'] = 'classification_regression'
if args['classification_num'] != 0 and args['regression_num'] == 0:
    args['task_class'] = 'classification'
if args['classification_num'] == 0 and args['regression_num'] != 0:
    args['task_class'] = 'regression'

args['bin_path'] = 'data/' + args['data_name'] + '.bin'
args['group_path'] = 'data/' + args['data_name'] + '_group.csv'

result_pd = pd.DataFrame(columns=args['select_task_list']+['group'] + args['select_task_list']+['group']
                         + args['select_task_list']+['group'])
all_times_train_result = []
all_times_val_result = []
all_times_test_result = []
for time_id in range(args['times']):
    set_random_seed(2020 + time_id)
    one_time_train_result = []
    one_time_val_result = []
    one_time_test_result = []
    print('***************************************************************************************************')
    print('{}, {}/{} time'.format(args['task_name'], time_id + 1, args['times']))
    print('***************************************************************************************************')
    train_set, val_set, test_set, task_number = build_dataset.load_graph_from_csv_bin_for_splited(
        bin_path=args['bin_path'],
        group_path=args['group_path'],
        select_task_index=args['select_task_index']
    )
    print("Molecule graph generation is complete !")
    train_loader = DataLoader(dataset=train_set,
                              batch_size=args['batch_size'],
                              shuffle=True,
                              collate_fn=collate_molgraphs)

    val_loader = DataLoader(dataset=val_set,
                            batch_size=args['batch_size'],
                            shuffle=True,
                            collate_fn=collate_molgraphs)

    test_loader = DataLoader(dataset=test_set,
                             batch_size=args['batch_size'],
                             collate_fn=collate_molgraphs)
    pos_weight_np = pos_weight(train_set, classification_num=args['classification_num'])
    loss_criterion_c = torch.nn.BCEWithLogitsLoss(reduction='none', pos_weight=pos_weight_np.to(args['device']))
    loss_criterion_r = torch.nn.MSELoss(reduction='none')
    model = MGA(in_feats=args['in_feats'], rgcn_hidden_feats=args['rgcn_hidden_feats'],
                   n_tasks=task_number, rgcn_drop_out=args['rgcn_drop_out'],
                   classifier_hidden_feats=args['classifier_hidden_feats'], dropout=args['drop_out'],
                   loop=args['loop'], return_weight=True)
    optimizer = Adam(model.parameters(), lr=10 ** -args['lr'], weight_decay=10 ** -args['weight_decay'])
    stopper = EarlyStopping(patience=args['patience'],
                            task_name=args['task_name'], mode=args['mode'])
    model.to(args['device'])
    stopper.load_checkpoint(model)
    # selected test molecules
    selected_data = pd.read_csv('data/KOWall4_group.csv')
    selected_mol_list = selected_data['smiles'].tolist()


    # visual test set
run_an_eval_epoch_heterogeneous_return_weight_py2(args, model, test_loader, vis_list=selected_mol_list, vis_task='BMF')
run_an_eval_epoch_heterogeneous_return_weight_py2(args, model, train_loader, vis_list=selected_mol_list, vis_task='BMF')
run_an_eval_epoch_heterogeneous_return_weight_py2(args, model, val_loader, vis_list=selected_mol_list, vis_task='BMF')


