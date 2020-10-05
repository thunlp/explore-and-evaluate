import time
import torch
import numpy as np
import random

def print_time_info(string, end='\n', dash_top=False, dash_bot=False, file=None):
    times = str(time.strftime('%Y-%m-%d %H:%M:%S',
                              time.localtime(time.time())))
    string = "[%s] %s" % (times, str(string))
    if dash_top:
        print(len(string) * '-', file=file)
    print(string, end=end, file=file)
    if dash_bot:
        print(len(string) * '-', file=file)


def set_random_seed(seed_value=0, print_info=False):
    if print_info:
        print_time_info('Random seed is set to %d.' % (seed_value))
    torch.manual_seed(seed_value)  # cpu  vars
    np.random.seed(seed_value)  # cpu vars
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    random.seed(seed_value)


def get_hits(sim, top_k=(1, 10), print_info=True, device='cpu'):
    if isinstance(sim, np.ndarray):
        sim = torch.from_numpy(sim)
    top_lr, mr_lr, mrr_lr = topk(sim, top_k, device=device)
    top_rl, mr_rl, mrr_rl = topk(sim.t(), top_k, device=device)

    if print_info:
        print_time_info('For each source:', dash_top=True)
        print_time_info('MR: %.2f; MRR: %.2f%%.' % (mr_lr, mrr_lr))
        for i in range(len(top_lr)):
            print_time_info('Hits@%d: %.2f%%' % (top_k[i], top_lr[i]))
        print('')
        print_time_info('For each target:')
        print_time_info('MR: %.2f; MRR: %.2f%%.' % (mr_rl, mrr_rl))
        for i in range(len(top_rl)):
            print_time_info('Hits@%d: %.2f%%' % (top_k[i], top_rl[i]))
        print_time_info('-------------------------------------')
    # return Hits@10
    return top_lr, top_rl, mr_lr, mr_rl, mrr_lr, mrr_rl


def topk(sim, top_k=(1, 10, 50, 100), device='cpu'):
    # Sim shape = [num_ent, num_ent]
    assert sim.shape[0] == sim.shape[1]
    test_num = sim.shape[0]
    batched = True
    if sim.shape[0] * sim.shape[1] < 20000 * 128:
        batched = False
        sim = sim.to(device)

    def _opti_topk(sim):
        sorted_arg = torch.argsort(sim)
        true_pos = torch.arange(test_num, device=device).reshape((-1, 1))
        locate = sorted_arg - true_pos
        del sorted_arg, true_pos
        locate = torch.nonzero(locate == 0)
        cols = locate[:, 1]  # Cols are ranks
        cols = cols.float()
        top_x = [0.0] * len(top_k)
        for i, k in enumerate(top_k):
            top_x[i] = float(torch.sum(cols < k)) / test_num * 100
        mr = float(torch.sum(cols + 1)) / test_num
        mrr = float(torch.sum(1.0 / (cols + 1))) / test_num * 100
        return top_x, mr, mrr

    def _opti_topk_batched(sim):
        mr = 0.0
        mrr = 0.0
        top_x = [0.0] * len(top_k)
        batch_size = 1024
        for i in range(0, test_num, batch_size):
            batch_sim = sim[i:i + batch_size].to(device)
            sorted_arg = torch.argsort(batch_sim)
            true_pos = torch.arange(
                batch_sim.shape[0]).reshape((-1, 1)).to(device) + i
            locate = sorted_arg - true_pos
            del sorted_arg, true_pos
            locate = torch.nonzero(locate == 0,)
            cols = locate[:, 1]  # Cols are ranks
            cols = cols.float()
            mr += float(torch.sum(cols + 1))
            mrr += float(torch.sum(1.0 / (cols + 1)))
            for i, k in enumerate(top_k):
                top_x[i] += float(torch.sum(cols < k))
        mr = mr / test_num
        mrr = mrr / test_num * 100
        for i in range(len(top_x)):
            top_x[i] = top_x[i] / test_num * 100
        return top_x, mr, mrr

    with torch.no_grad():
        if not batched:
            return _opti_topk(sim)
        return _opti_topk_batched(sim)



