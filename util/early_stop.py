import numpy as np
import torch

class EarlyStopMonitor(object):
    def __init__(self, save_dir, max_round=3, higher_better=True, tolerance=1e-10):
        self.max_round = max_round
        self.num_round = 0

        self.save_dir = save_dir

        self.best_model_file = None
        self.best_optimizer_file = None
        self.user_perf_file = None
        self.user_knowl_file = None
        self.item_perf_file = None
        self.item_knowl_file = None
        self.best_train_file = None
        self.best_valid_file = None

        self.last_best = None
        self.best_mrr = None
        self.higher_better = higher_better
        self.tolerance = tolerance

    def early_stop_check(self, curr_val, cur_mrr, model_state, 
                         user_perf=None, user_knowl=None, item_perf=None, item_knowl=None,
                         optimizer=None, train_graph=None, valid_graph=None, chunk_index=None):
        if not self.higher_better:
            curr_val *= -1
        if self.last_best is None:
            self.last_best = curr_val
            self.best_mrr = cur_mrr
            if chunk_index is None:
                self.best_model_file = f'{self.save_dir}/model_state.pt'
                self.best_optimizer_file = f'{self.save_dir}/optimizer_state.pt'
                self.best_train_file = f'{self.save_dir}/best_train_graph.pt'
                self.best_valid_file = f'{self.save_dir}/best_valid_graph.pt'
            else:
                self.best_model_file = f'{self.save_dir}/chunk{chunk_index}_model_state.pt'
                self.best_optimizer_file = f'{self.save_dir}/chunk{chunk_index}_optimizer_state.pt'
                self.user_perf_file = f'{self.save_dir}/chunk{chunk_index}_user_perf.pt'
                self.user_knowl_file = f'{self.save_dir}/chunk{chunk_index}_user_knowl.pt'
                self.item_perf_file = f'{self.save_dir}/chunk{chunk_index}_item_perf.pt'
                self.item_knowl_file = f'{self.save_dir}/chunk{chunk_index}_item_knowl.pt'
                self.best_train_file = f'{self.save_dir}/chunk{chunk_index}_train_graph.pt'
                self.best_valid_file = f'{self.save_dir}/chunk{chunk_index}_valid_graph.pt'
            if model_state is not None:
                torch.save(model_state, self.best_model_file)
            if user_perf is not None:
                torch.save(user_perf, self.user_perf_file)
                torch.save(user_knowl, self.user_knowl_file)
                torch.save(item_perf, self.item_perf_file)
                torch.save(item_knowl, self.item_knowl_file)
            if optimizer is not None:
                torch.save(optimizer, self.best_optimizer_file)
            if train_graph is not None:
                torch.save(train_graph, self.best_train_file)
            if valid_graph is not None:
                torch.save(valid_graph, self.best_valid_file)
        elif (curr_val - self.last_best) / np.abs(self.last_best) > self.tolerance:
            self.last_best = curr_val
            self.num_round = 0
            if cur_mrr > self.best_mrr:
                self.best_mrr = cur_mrr
                if model_state is not None:
                    torch.save(model_state, self.best_model_file)
                if user_perf is not None:
                    torch.save(user_perf, self.user_perf_file)
                    torch.save(user_knowl, self.user_knowl_file)
                    torch.save(item_perf, self.item_perf_file)
                    torch.save(item_knowl, self.item_knowl_file)
                if optimizer is not None:
                    torch.save(optimizer, self.best_optimizer_file)
                if train_graph is not None:
                    torch.save(train_graph, self.best_train_file)
                if valid_graph is not None:
                    torch.save(valid_graph, self.best_valid_file)
        else:
            self.num_round += 1

        return self.num_round >= self.max_round