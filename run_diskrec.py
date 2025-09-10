import torch
from torch import nn
from tqdm import tqdm
from torch.optim import Adam
import argparse, copy

import os, sys, time
cur_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(cur_path[:cur_path.find('DISKRec-SIGIR')] + 'DISKRec-SIGIR') 

from model.diskrec import DISKRec
from evaluation.metric import env
from model.graph import GraphData
from util.data_util import load_uk_interaction, load_target_rating, load_target_negative, save_live, average_state_dict
from util.early_stop import EarlyStopMonitor
from util.global_config import setup_seed, get_datadir


parser = argparse.ArgumentParser(description = 'Show description')
parser.add_argument('--model', type=str, default='DISKRec', help='model name')
parser.add_argument('-d', '--dataset', type=str, default='mooccubex', help='dataset')
parser.add_argument('-cuda', '--cuda', type=int, default=0, help='gpu id')
parser.add_argument('-seed', '--seed', type=int, default=123, help='seed')
parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3, help='learning rate')
parser.add_argument('-l2', '--l2', type=float, default=5e-4, help='the weight of L2 Regularization')
parser.add_argument('-epoch', '--epochs', type=int, default=500, help='epochs')
parser.add_argument('-early', '--early', type=bool, default=True, help='whether early stop')
parser.add_argument('-pat', '--patience', type=int, default=20, help='the patience of early stop')
parser.add_argument('-alpha', '--alpha', type= float, default=0.5, help='moving average factor')

parser.add_argument('-nocross', '--cross', action='store_false', help='whether to cross update two status')
parser.add_argument('-nodecay', '--decay', action='store_false', help='whether to decay cell')
parser.add_argument('-kt', '--kt', type=str, default='nokt', help='kt net')
parser.add_argument('-beta', '--beta', type=float, default=1, help='the weight of kt loss')
parser.add_argument('-st', '--state_mode', type=str, default='both', help='both, knowl, pref')
parser.add_argument('-nl', '--nei_algo', type=str, default='last', help='neighbor Algorithm')
parser.add_argument('-notrans', '--trans_message', action='store_false', help='whether to transform message')
parser.add_argument('-msg_agg', '--msg_agg', type=str, default='mean', help='message aggregation method')
parser.add_argument('-fus', '--fusion', type=str, default='linear', help='fusion function')
parser.add_argument('-nd', '--node_dims', type=int, default=100, help='the dimension of graph node')
parser.add_argument('-md', '--message_dims', type=int, default=512, help='the dimension of message')
parser.add_argument('-hd', '--hidden_dims', type=int, default=256, help='the dimension of hidden state and cell memory')
parser.add_argument('-ld', '--latent_dim', type=int, default=40, help='the dimension of latent factor')
parser.add_argument('-w', '--w', type=float, default=1.1, help='the weight of decay method')
parser.add_argument('-dm', '--decay_method', type=str, default='log', help='decay method')
parser.add_argument('-act', '--act', type=str, default='tanh', help='activation function')
parser.add_argument('-bs', '--batch_size', type=int, default=1024, help='batch_size')
parser.add_argument('-ns', '--neighbors', type=int, default=10, help='the neighbor number')

args = parser.parse_args()

setup_seed(args.seed)
device = torch.device(f'cuda:{args.cuda}') if torch.cuda.is_available() else "cpu"
torch.cuda.set_device(device)

if args.state_mode in ['pref', 'knowl', 'one']: args.cross = False
if not args.trans_message: args.message_dim = args.hidden_dims
print(args)

current_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
input_dir = f'{get_datadir()}/model_input/{args.dataset}'
output_dir = f'{get_datadir()}/model_output/{args.dataset}/live_update'
cross_flag = 'cross' if args.cross else 'nocross'
decay_flag = 'decay' if args.decay else 'nodecay'
kt_flag = args.kt
mes_flag = 'TransW' if args.trans_message else 'noTransW'
save_dir = f'{output_dir}/DISKRec_{cross_flag}_{decay_flag}_{kt_flag}_b{args.beta}_{mes_flag}_{args.state_mode}_{args.nei_algo}_{args.msg_agg}_{args.fusion}_b{args.batch_size}_n{args.neighbors}_a{args.alpha}'
os.makedirs(save_dir, exist_ok=True)
print(f'save_dir: {save_dir}')
config_file = f'{save_dir}/model_config_{args.seed}.txt'
with open(config_file, 'w') as f:
    f.write(str(args))

rating = load_target_rating(input_dir, 0, device=device)
node_nums = {'user':rating.shape[0], 'concept':rating.shape[1]}
model = DISKRec(node_nums, args=args, device=device).to(device)

model_init = None
test_metric = []

def live_update(train_graph=None, chunk_index=0, 
                train_data=None, train_target_rating=None, train_negative=None,
                valid_data=None, valid_target_rating=None, valid_negative=None,
                test_data=None, test_target_rating=None, test_negative=None):
    start_time = time.time()
    
    global model_init, model, train_metric, valid_metric, test_metric
    if model_init is not None: model.load_state_dict(copy.deepcopy(model_init))
    optimizer = Adam(params=model.parameters(), lr=args.learning_rate, weight_decay=args.l2)

    early_stop_monitor = EarlyStopMonitor(save_dir, max_round=args.patience, higher_better=False, tolerance=1e-10)
    train_metric, valid_metric, GPU_usage = [], [], []
    epoch_bar = tqdm([i for i in range(args.epochs)], desc='Epoch = xx [ loss: xx]', position=2, leave=False)
    for epoch in epoch_bar:
        model.train()
        model.dgnn.reset_params()
        gpu1 = torch.cuda.memory_allocated(device) / 1024**3
        result = model(train_graph, train_data, train_target_rating)
        gpu2 = torch.cuda.memory_allocated(device) / 1024**3
        train_rating = result['rating']
        train_loss = result['loss']
        train_out_graph = result['out_graph']
        train_node_timestampe, train_neighbors = model.dgnn.get_params()

        optimizer.zero_grad()
        train_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        train_h5, train_h10, train_n5, train_n10, train_mrr = env(train_rating, train_negative)
        train_metric.append([epoch, train_loss.item(), train_h5, train_h10, train_n5, train_n10, train_mrr])
        GPU_usage.append([epoch, gpu1, gpu2])
        
        model.eval()
        with torch.no_grad():
            result = model(train_out_graph, valid_data, valid_target_rating)
            valid_rating = result['rating']
            valid_loss = result['loss']
            valid_out_graph = result['out_graph']
        valid_h5, valid_h10, valid_n5, valid_n10, valid_mrr = env(valid_rating, valid_negative)
        valid_metric.append([epoch, valid_loss.item(), valid_h5, valid_h10, valid_n5, valid_n10, valid_mrr])

        if args.early:
            is_early = early_stop_monitor.early_stop_check(valid_loss.item(), valid_mrr, chunk_index=chunk_index, model_state=model.state_dict(),
                                                           train_graph=train_out_graph, valid_graph=valid_out_graph)
            if is_early: break 
        epoch_bar.set_description(f'Epoch = {epoch} [ pat={early_stop_monitor.num_round}, gpu: {gpu1:.2f}/{gpu2:.2f}, lr={optimizer.param_groups[0]["lr"]}, '\
                                f'loss: {train_loss:.4f}/{valid_loss:.4f}, h5: {train_h5:.2f}/{valid_h5:.2f}, n5: {train_n5:.2f}/{valid_n5:.2f}]')
    
    model.eval()
    if args.early:
        model.load_state_dict(torch.load(early_stop_monitor.best_model_file))
        valid_out_graph = torch.load(early_stop_monitor.best_valid_file, map_location=device)
    with torch.no_grad():
        result = model(valid_out_graph, test_data, test_target_rating)
        test_rating = result['rating']
        test_loss = result['loss']
    test_h5, test_h10, test_n5, test_n10, test_mrr = env(test_rating, test_negative)
    end_time = time.time()
    test_metric.append([epoch, test_loss.item(), test_h5, test_h10, test_n5, test_n10, test_mrr, end_time-start_time])
    
    if model_init is None:
        model_init = copy.deepcopy(model.state_dict())
    else:
        model_init = average_state_dict(model_init, model.state_dict(), args.alpha)
    
    model.dgnn.update_params(train_node_timestampe, train_neighbors)

    save_live(save_dir=save_dir, chunk_index=chunk_index, GPU_usage=GPU_usage, seed=args.seed,
              train_rating=train_rating, valid_rating=valid_rating, test_rating=test_rating, 
              train_metric=train_metric, valid_metric=valid_metric, test_metric=test_metric)
    return torch.load(early_stop_monitor.best_train_file, map_location=device)

if args.state_mode in ['pref']:
    graph = GraphData(node_nums, args.node_dims, args.hidden_dims, has_pref=True, device=device)
elif args.state_mode=='knowl':
    graph = GraphData(node_nums, args.node_dims, args.hidden_dims, has_knowl=True, device=device)
else:
    graph = GraphData(node_nums, args.node_dims, args.hidden_dims, has_pref=True, has_knowl=True, device=device)
graph.generate_graph()
input_graph = graph.get_graph()

chunk_bar = tqdm([i for i in range(0, 8)], desc='Chunk = xx [ loss: xx/xx ]', position=1, leave=False)
for chunk_index in chunk_bar:
    train_data = load_uk_interaction(input_dir, chunk_index, device=device)
    train_target_rating = load_target_rating(input_dir, chunk_index, device=device)
    train_negative = load_target_negative(input_dir, chunk_index, device=device)

    valid_data = load_uk_interaction(input_dir, chunk_index+1, device=device)
    valid_target_rating = load_target_rating(input_dir, chunk_index+1, device=device)
    valid_negative = load_target_negative(input_dir, chunk_index+1, device=device)

    test_data = load_uk_interaction(input_dir, chunk_index+2, device=device)
    test_target_rating = load_target_rating(input_dir, chunk_index+2, device=device)
    test_negative = load_target_negative(input_dir, chunk_index+2, device=device)

    train_out_graph = live_update(train_graph=input_graph, chunk_index=chunk_index, 
                                    train_data=train_data, train_target_rating=train_target_rating, train_negative=train_negative,
                                    valid_data=valid_data, valid_target_rating=valid_target_rating, valid_negative=valid_negative,
                                    test_data=test_data, test_target_rating=test_target_rating, test_negative=test_negative)
    input_graph = train_out_graph

    chunk_bar.set_description(f'Chunk = {chunk_index+1} [ loss: {test_metric[-1][1]:.4f}, HR@5: {test_metric[-1][2]:.2f}, HR@10: {test_metric[-1][3]:.2f}, '\
                            + f'NDCG@5: {test_metric[-1][4]:.2f}, NDCG@10: {test_metric[-1][5]:.2f}, MRR: {test_metric[-1][6]:.2f} ]')

print(f'\nfinal last metric: loss: {test_metric[-1][1]:.4f}, HR@5: {test_metric[-1][2]:.4f}, HR@10: {test_metric[-1][3]:.4f}, '\
      f'NDCG@5: {test_metric[-1][4]:.4f}, NDCG@10: {test_metric[-1][5]:.4f}, MRR: {test_metric[-1][6]:.4f}\n')    
