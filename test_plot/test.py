from utils.functions import *
from test_plot.test_dataset import Test_CSPDataset, render, tour_len_dist, clean_tour
from torch.utils.data import DataLoader



def LS(loc, tour=None, cover_range=7, tag="RL"):
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
# 对比：attention2019。
# model_path = "/home/kevin/PycharmProjects/CSP_Attention/outputs/csp_100/run_20191204T111536/epoch-3.pt"
# 4heads
# model_path = "/home/kevin/PycharmProjects/CSP_Attention/outputs/csp_100/run_20191204T140251/epoch-6.pt"
# 4heads. cat(rnn_out , fixed.context)
# model_path = "/home/kevin/PycharmProjects/CSP_Attention/outputs/csp_100/run_20191209T100023/epoch-7.pt"
# 8 heads cat(rnn_out , fixed.context)
# model_path = "/home/kevin/PycharmProjects/CSP_Attention/outputs/csp_100/run_20191209T110554/epoch-4.pt"
# query = last_hh
# model_path = "/home/kevin/PycharmProjects/CSP_Attention/outputs/csp_100/run_20191209T110554/epoch-22.pt"
# model_path = "/home/kevin/PycharmProjects/CSP_Attention/outputs/csp_100/run_20200617T111940/epoch-29.pt"
# center_distance
# model_path = "/home/kevin/PycharmProjects/CSP_Attention/outputs/csp_100/run_20200618T111933/epoch-34.pt"
# can be resumed: K.mul(1+dynamic) ,bias=False, len+num, lr_decay
# /home/kevin/PycharmProjects/CSP_Attention/outputs/csp_100/run_20200622T091539
# clip = 0, len
model_path = "/home/kevin/PycharmProjects/CSP_Attention/outputs/csp_100/run_20200622T103952/epoch-29.pt"


model, _ = load_model(model_path)

cover_range = 7
num_sample = 50
# dataset = Test_CSPDataset(size=150, num_samples=num_sample, cover_range=cover_range, seed=12)
# dataset = Test_CSPDataset(size=52, num_samples=10, seed=13, tsp_name='berlin52')
# dataset = Test_CSPDataset(size=60, num_samples=1, seed=13, tsp_name='kroA150')

# dataloader = DataLoader(dataset, batch_size=num_sample)

model.to(device)
model.eval()
model.set_decode_type("greedy")



# import time
# for batch in dataloader:
#     batch = move_to(batch, device)
#     with torch.no_grad():
#         t1=time.time()
#         # cost, logp, tour = model(batch, return_pi=True)
#
#         cum_log_p, sequences, costs, ids, batch_size = model.beam_search(
#             batch, beam_size=2
#         )
#         print(sequences)
#         print(costs)
#         ids = torch.arange(100).unsqueeze(1).view(-1,2)
#
#         dists,len = tour_len_dist(batch, sequences[ids[:,0],:])
#         print(time.time()-t1)
#         print(dists)
#         print(np.mean(len))
#         print(dists.mean())
#
#
# import time
# for batch in dataloader:
#     batch = move_to(batch, device)
#     with torch.no_grad():
#         t1=time.time()
#         cost, logp, tour = model(batch, return_pi=True)
#         dists,len = tour_len_dist(batch, tour)
#         print(time.time()-t1)
#         print(dists)
#         print(np.mean(len))
#         print(dists.mean())
#
#         loc = batch['loc'].transpose(-1, -2)
#         render(loc,tour,os.path.join('plot_dir', 'test_%2.3f.png'%(cost.mean().item())))
# #
#         # loc = batch.squeeze(0).transpose(0, 1).cpu().numpy()
#         # print(tour.size())
#         # tour = tour.cpu().numpy()
#         # final_tour, dists = LS(batch, tour, cover_range, tag='LS1')
# #
# #
#
# # loc = loc.squeeze(0).transpose(0, 1).cpu().numpy()
# index = 0
# loc = loc[index, :, :].squeeze(0).transpose(0, 1).cpu().numpy()
# tour = tour[index, :].cpu().numpy()
# print(loc.shape, tour.shape)

import time
cover_range = 7
num_sample = 1000

c_rl = []
c_ls = []
for i in range(1):
    # seed 32, 12 (max 300)  3552
    dataset = Test_CSPDataset(size=100, num_samples=num_sample, cover_range=cover_range, seed=123, sample_mode=True)
    dataloader = DataLoader(dataset, batch_size=num_sample)

    model.set_decode_type("sampling")

    for batch in dataloader:
        batch = move_to(batch, device)
        with torch.no_grad():
            t1 = time.time()
            print(batch['loc'].size())
            cost, logp, tour = model(batch, return_pi=True)
            dists, len = tour_len_dist(batch, tour)
            print(time.time() - t1)
            sorted, inds = dists.sort()
            print(sorted[:9])
            loc = batch['loc'].transpose(-1, -2)

            render(loc[inds[:9]], tour[inds[:9]], os.path.join('plot_dir', 'test_%2.3f.png' % (cost.mean().item())))

            c_rl.append(dists.min().cpu().numpy())
            index = dists.argmin()
            loc = loc[index, :, :].squeeze(0).transpose(0, 1).cpu().numpy()
            tour = tour[index, :].cpu().numpy()

            tour = clean_tour(tour)

            from LS.LS1 import dist

            opt_path, dists1 = LS(loc, tour=tour, cover_range=cover_range, tag='RL')
            print("RL,afer_ls:", dist(loc, tour), dist(loc, opt_path))
    #         # loc = batch.squeeze(0).transpose(0, 1).cpu().numpy()
    #         # print(tour.size())
    #         # tour = tour.cpu().numpy()



    final_tour, dists_LS = LS(loc, tour=None, cover_range=cover_range, tag='LS1')
    c_ls.append(dists_LS)


print("LS:", np.array(c_ls).mean())
print("RL:", np.array(c_rl).mean())
print("LS:", np.array(c_ls))
print("RL:", np.array(c_rl))

