from utils.functions import *
from test_plot.test_dataset import Test_CSPDataset, render, tour_len_dist
from torch.utils.data import DataLoader

def LS(batch, tour=None, cover_range=7, tag="RL"):
    t2 = time.time()
    if tag=="LS1":
        from LS.LS1 import LS, render, dist
    elif tag=="RL":
        from LS.LS_RL import LS, render, dist

    final_tour = LS(loc, cover_range, tour)
    dists = dist(loc, final_tour)
    render(loc, final_tour, cover_range)

    print("LS time:", time.time() - t2)
    print("After LS")
    print(final_tour)
    print("Cost:", dists)
    return final_tour, dists

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# STP
model_path = "/home/kevin/PycharmProjects/CSP_Attention/outputs/csp_50/run_20191109T222443/epoch-29.pt"
# x0,gru
model_path = "/home/kevin/PycharmProjects/CSP_Attention/outputs/csp_50/run_20191107T175019/epoch-20.pt"
# 100 nodes, 256 batch, x0
model_path = "/home/kevin/PycharmProjects/CSP_Attention/outputs/csp_100/run_20191112T165220/epoch-5.pt"
# 100 nodes, 256 batch, first nn.Parameter
model_path = "/home/kevin/PycharmProjects/CSP_Attention/outputs/csp_100/run_20191112T171535/epoch-0.pt"
# model_path = "/home/kevin/PycharmProjects/CSP_Attention/outputs/csp_100/run_20191112T200626/epoch-10.pt"
# log cover...Now the best
model_path = "/home/kevin/PycharmProjects/CSP_Attention/outputs/csp_100/run_20191114T182228/epoch-0.pt"
# # critic
# model_path = "/home/kevin/PycharmProjects/CSP_Attention/outputs/csp_100/run_20191118T090332/epoch-5.pt"
# # bias=false project
# model_path = "/home/kevin/PycharmProjects/CSP_Attention/outputs/csp_100/run_20191118T111301/epoch-0.pt"
# one layer
# model_path = "/home/kevin/PycharmProjects/CSP_Attention/outputs/csp_100/run_20191118T160424/epoch-0.pt"
# original first, rollout, 0-6/7. Now the best. 7.38,10.9 epoch-1   7.3,11.48 EPOCH3
model_path = "/home/kevin/PycharmProjects/CSP_Attention/outputs/csp_100/run_20191118T172011/epoch-3.pt"
# nums+len
# model_path = "/home/kevin/PycharmProjects/CSP_Attention/outputs/csp_100/run_20191119T144746/epoch-29.pt"
model, _ = load_model(model_path)

cover_range = 7
dataset = Test_CSPDataset(size=400, num_samples=200, cover_range=cover_range, seed=123)
# dataset = Test_CSPDataset(size=52, num_samples=10, seed=13, tsp_name='berlin52')
# dataset = Test_CSPDataset(size=60, num_samples=1, seed=13, tsp_name='kroA150')

dataloader = DataLoader(dataset, batch_size=200)

model.to(device)
model.eval()
model.set_decode_type("greedy")

import time
for batch in dataloader:
    batch = move_to(batch, device)
    with torch.no_grad():
        t1=time.time()
        cost, logp, tour = model(batch, return_pi=True)
        dists,len = tour_len_dist(batch, tour)
        print(time.time()-t1)
        print(dists)
        print(np.mean(len))
        print(dists.mean())


        batch = batch['loc'].transpose(-1,-2)
        render(batch,tour,os.path.join('plot_dir', 'test_%2.3f.png'%(cost.mean().item())))

        loc = batch.squeeze(0).transpose(0, 1).cpu().numpy()
        tour = tour.cpu().numpy().squeeze(0)
        final_tour, dists = LS(batch, tour, cover_range, tag='LS1')


compare=True
if compare:
    print('\r\n'*3)
    final_tour, dists = LS(batch, tour=None, cover_range=cover_range, tag='LS1')
