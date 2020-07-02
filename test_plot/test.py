from utils.functions import *
from test_plot.test_dataset import Test_CSPDataset, render, tour_len_dist, clean_tour
from torch.utils.data import DataLoader

def LS(loc, tour=None, cover_range=7, tag="RL", render_if = True, prinf_if = True):
    t2 = time.time()
    if tag=="LS1":
        from LS.LS1 import LS, render, dist
    elif tag=="RL":
        from LS.LS_RL import LS, render, dist
    elif tag == "LS2":
        from LS.LS2 import LS, render, dist

    final_tour = LS(loc, cover_range, tour, prinf_if)
    dists = dist(loc, final_tour)
    if render_if:
        render(loc, final_tour, cover_range)

    print("LS time:", time.time() - t2)
    print("Cost:", dists)
    return final_tour, dists, time.time() - t2

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# original first, rollout, 0-6/7. Now the best. 7.38,10.9 epoch-1   7.3,11.48 EPOCH3
model_path = "/home/kevin/PycharmProjects/CSP_Attention/outputs/csp_100/run_20191118T172011/epoch-3.pt"
# nums+len
# model_path = "/home/kevin/PycharmProjects/CSP_Attention/outputs/csp_100/run_20191119T144746/epoch-29.pt"
# 对比：attention2019。
# model_path = "/home/kevin/PycharmProjects/CSP_Attention/outputs/csp_100/run_20191204T111536/epoch-3.pt"
model_path = "/home/kevin/PycharmProjects/CSP_Attention/outputs/csp_100/run_20200617T111940/epoch-29.pt"
# center_distance
# model_path = "/home/kevin/PycharmProjects/CSP_Attention/outputs/csp_100/run_20200618T111933/epoch-34.pt"
# can be resumed: K.mul(1+dynamic) ,bias=False, len+num, lr_decay
# /home/kevin/PycharmProjects/CSP_Attention/outputs/csp_100/run_20200622T091539
# 100,clip = 0, len. BEST, 256batch.160000epoch.4,17
model_path = "/home/kevin/PycharmProjects/CSP_Attention/outputs/csp_100/run_20200622T103952/epoch-29.pt"
# model_path = "/home/kevin/PycharmProjects/CSP_Attention/outputs/csp_100/run_20200622T145808/epoch-56.pt"
# 50,clip = 0. BEST, 512batch.640000epoch
model_path = '/home/kevin/PycharmProjects/CSP_Attention/outputs/csp_50/run_20200629T110102/epoch-31.pt'
# 384batch 384000epoch.4.07,epoch-30
model_path = '/home/kevin/PycharmProjects/CSP_Attention/outputs/csp_100/run_20200629T210739/epoch-30.pt'

model_path = '/home/kevin/PycharmProjects/CSP_Attention/outputs/csp_200/run_20200630T170000/epoch-49.pt'
# model.set_decode_type("greedy")
import multiprocessing
from multiprocessing import Pool
import time


def LS_async(loc, cover_range, c_ls, t_ls, c_ls2, t_ls2):
    final_tour, dists_LS, t = LS(loc, tour=None, cover_range=cover_range, tag='LS1',
                              render_if=False, prinf_if=False)
    c_ls.append(dists_LS)
    t_ls.append(t)

    final_tour, dists_LS2, t2 = LS(loc, tour=None, cover_range=cover_range, tag='LS2',
                              render_if=False, prinf_if=False)
    c_ls2.append(dists_LS2)
    t_ls2.append(t2)

def RL_LS_async(loc, tour, cover_range, c_ls, t_ls):
    final_tour, dists_LS, t = LS(loc, tour=tour, cover_range=cover_range, tag='RL',
                              render_if=False, prinf_if=False)
    c_ls.append(dists_LS)
    t_ls.append(t)

def test_batch(test_LS=True, test_RL=True):

    if cities <= 50:
        model_path = '/home/kevin/PycharmProjects/CSP_Attention/outputs/csp_50/run_20200629T110102/epoch-31.pt'
    elif cities <= 150:
        model_path = '/home/kevin/PycharmProjects/CSP_Attention/outputs/csp_100/run_20200629T210739/epoch-30.pt'
    else:
        model_path = '/home/kevin/PycharmProjects/CSP_Attention/outputs/csp_200/run_20200630T170000/epoch-49.pt'
    model, _ = load_model(model_path)
    model.to(device)
    model.eval()

    test_nums = 100
    # 123: 7, 100city, LS(3.948),RL(4.19/4.06),ls-400
    # 123: 11, 100city, LS(3.118), RL(),ls-400
    # 123：7, 20citiy, LS(2.52), RL(2.15),ls,400
    # 123：7, 200citiy, LS(6.20), RL(6.78/6.98),ls,400
    # 123：7, 150citiy, LS(5.075), RL(6.78/5.12),ls,400
    seed_seed = 123456
    np.random.seed(seed_seed)
    seeds = np.random.randint(0, 123456, test_nums)
    print("Seeds:", seeds)

    if test_LS:
        # Multiprocessing to run LS
        pool = Pool()
        manager = multiprocessing.Manager()
        # save the cost and time
        c_ls = manager.list()
        t_ls = manager.list()
        c_ls2 = manager.list()
        t_ls2 = manager.list()

        for i in range(test_nums):
            dataset = Test_CSPDataset(size=cities, num_samples=1, cover_range=cover_range, seed=seeds[i],
                                      sample_mode=True)
            dataloader = DataLoader(dataset, batch_size=1)
            for batch in dataloader:
                loc = batch['loc'].transpose(-1, -2)
                loc = loc[0, :, :].squeeze(0).transpose(0, 1).cpu().numpy()
                # worker
                pool.apply_async(LS_async, args=(loc, cover_range, c_ls, t_ls, c_ls2, t_ls2,))

        pool.close()
        pool.join()
        print("LS1_time:", np.mean(list(t_ls)))
        print("LS1_cost:", np.mean(list(c_ls)))
        print("LS2_time:", np.mean(list(t_ls2)))
        print("LS2_cost:", np.mean(list(c_ls2)))


    c_rl = []
    loc_list = []
    tour_list = []
    # First obtain the RL results by sampling.
    for i in range(test_nums):
        dataset = Test_CSPDataset(size=cities, num_samples=num_sample, cover_range=cover_range, seed=seeds[i],
                                  sample_mode=True)
        dataloader = DataLoader(dataset, batch_size=num_sample)
        model.set_decode_type("sampling")

        for batch in dataloader:
            batch = move_to(batch, device)
            with torch.no_grad():
                cost, logp, tour = model(batch, return_pi=True)
                dists, len = tour_len_dist(batch, tour)
                loc = batch['loc'].transpose(-1, -2)
                c_rl.append(dists.min().cpu().numpy())
                index = dists.argmin()
                loc = loc[index, :, :].squeeze(0).transpose(0, 1).cpu().numpy()
                tour = tour[index, :].cpu().numpy()
                tour = clean_tour(tour)

        loc_list.append(loc)
        tour_list.append(tour)

    # Then LS the RL results by Multiprocessing
    pool2 = Pool()
    manager = multiprocessing.Manager()
    c_rl_ls = manager.list()
    t_rl_ls = manager.list()

    for loc, tour in zip(loc_list, tour_list):
        pool2.apply_async(RL_LS_async, args=(loc, tour, cover_range, c_rl_ls, t_rl_ls,))
    pool2.close()
    pool2.join()

    print("RL:", np.array(c_rl).mean())

    print("RL_LS_time:", np.mean(list(t_rl_ls)))
    print("RL_LS_cost:", np.mean(list(c_rl_ls)))

    if test_LS:
        print("LS1_time:", np.mean(list(t_ls)))
        print("LS1_cost:", np.mean(list(c_ls)))
        print("LS2_time:", np.mean(list(t_ls2)))
        print("LS2_cost:", np.mean(list(c_ls2)))


def test_plot():
    model, _ = load_model(model_path)
    model.to(device)
    model.eval()

    dataset = Test_CSPDataset(size=cities, num_samples=num_sample,
                              cover_range=cover_range, seed=1234,
                              sample_mode=True)
    dataloader = DataLoader(dataset, batch_size=num_sample)

    model.set_decode_type("sampling")

    for batch in dataloader:
        batch = move_to(batch, device)
        with torch.no_grad():
            t1 = time.time()
            cost, logp, tour = model(batch, return_pi=True)
            dists, len = tour_len_dist(batch, tour)
            # print(time.time() - t1)
            sorted, inds = dists.sort()
            print("RL:", dists.min())
            loc = batch['loc'].transpose(-1, -2)

            render(loc[inds[:9]], tour[inds[:9]], os.path.join('plot_dir', 'test_%2.3f.png' % (cost.mean().item())))

            index = dists.argmin()
            loc = loc[index, :, :].squeeze(0).transpose(0, 1).cpu().numpy()
            tour = tour[index, :].cpu().numpy()

            #
            tour = clean_tour(tour)
            from LS.LS1 import dist
            from LS.LS_2opt import LS_2opt
            print("RL:", dist(loc, tour))
            #
            # opt_path = LS_2opt(loc, tour, 200)

            opt_path, dists1, _ = LS(loc, tour=tour, cover_range=cover_range, tag='RL')
            print("RL, afer_ls:", dist(loc, tour), dist(loc, opt_path))
    final_tour, dists_LS, _ = LS(loc, tour=None, cover_range=cover_range, tag='LS1')
    print("LS", dists_LS)

    final_tour, dists_LS2, _ = LS(loc, tour=None, cover_range=cover_range, tag='LS2')
    print("LS", dists_LS2)


if __name__ == '__main__':
    cover_range = 7
    num_sample = 1000
    cities = 200

    test_batch(test_LS=True)
    # test_plot()

