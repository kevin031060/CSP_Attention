from utils.functions import *
from test_plot.test_dataset import Test_CSPDataset, render, tour_len_dist, clean_tour
from torch.utils.data import DataLoader
# import matplotlib
# matplotlib.use('Agg')

def LS(loc, tour=None, cover_range=7, tag="RL", render_if = True, prinf_if = True, stop_cost=-1e6, radius = None):
    t2 = time.time()
    if tag == "LS1":
        from LS.LS1 import LS, render, dist
    elif tag == "RL":
        from LS.LS_RL import LS, render, dist
    elif tag == "LS2":
        from LS.LS2 import LS, render, dist

    final_tour = LS(loc, cover_range, tour, prinf_if, stop_cost, radius)
    dists = dist(loc, final_tour)
    if render_if:
        render(loc, final_tour, cover_range, radius)

    print("LS--", tag, " time:", time.time() - t2)
    print("LS--", tag, "Cost:", dists)
    return final_tour, dists, time.time() - t2

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# model.set_decode_type("greedy")
import multiprocessing
from multiprocessing import Pool
import time


# def LS_async(loc, cover_range, c_ls, t_ls, c_ls2, t_ls2, stop_cost):
#     final_tour, dists_LS, t = LS(loc, tour=None, cover_range=cover_range, tag='LS1',
#                               render_if=False, prinf_if=False, stop_cost=stop_cost)
#     c_ls.append(dists_LS)
#     t_ls.append(t)
#
#     final_tour, dists_LS2, t2 = LS(loc, tour=None, cover_range=cover_range, tag='LS2',
#                               render_if=False, prinf_if=False, stop_cost=stop_cost)
#     c_ls2.append(dists_LS2)
#     t_ls2.append(t2)
#
# def RL_LS_async(loc, tour, cover_range, c_ls, t_ls):
#     final_tour, dists_LS, t = LS(loc, tour=tour, cover_range=cover_range, tag='RL',
#                               render_if=False, prinf_if=False)
#     c_ls.append(dists_LS)
#     t_ls.append(t)
def LS_async(loc, cover_range, c_ls, t_ls, c_ls2, t_ls2, stop_cost=-1e6):
    final_tour, dists_LS, t = LS(loc, tour=None, cover_range=cover_range, tag='LS1',
                              render_if=False, prinf_if=False, stop_cost=stop_cost)
    c_ls.append(dists_LS)
    t_ls.append(t)

    final_tour, dists_LS2, t2 = LS(loc, tour=None, cover_range=cover_range, tag='LS2',
                              render_if=False, prinf_if=False, stop_cost=stop_cost)
    c_ls2.append(dists_LS2)
    t_ls2.append(t2)


def RL_LS_async(loc, tour, cover_range, c_ls, t_ls):
    final_tour, dists_LS, t = LS(loc, tour=tour, cover_range=cover_range, tag='RL',
                              render_if=False, prinf_if=False)
    c_ls.append([dists_LS,loc[0,0]])
    # c_ls.append(dists_LS)
    t_ls.append(t)

def LS_async_radius(loc, cover_range, c_ls, t_ls, c_ls2, t_ls2, radius, stop_cost=-1e6):
    final_tour, dists_LS, t = LS(loc, tour=None, cover_range=cover_range, tag='LS1',
                                 render_if=False, prinf_if=False, stop_cost=stop_cost, radius=radius)
    c_ls.append(dists_LS)
    t_ls.append(t)

    final_tour, dists_LS2, t2 = LS(loc, tour=None, cover_range=cover_range, tag='LS2',
                                   render_if=False, prinf_if=False, stop_cost=stop_cost, radius=radius)
    c_ls2.append(dists_LS2)
    t_ls2.append(t2)


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

    test_nums = 5

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
                pool.apply_async(LS_async, args=(loc, cover_range, c_ls, t_ls, c_ls2, t_ls2, 1e-6, ))

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
    stop_costs = np.array(list(c_rl_ls))
    print("RL_LS_cost:", np.mean(stop_costs[:,0]))


    if test_LS:
        print("LS1_time:", np.mean(list(t_ls)))
        print("LS1_cost:", np.mean(list(c_ls)))
        print("LS2_time:", np.mean(list(t_ls2)))
        print("LS2_cost:", np.mean(list(c_ls2)))

#     save
    with open(batch_results_file_path, 'a+') as f:
        f.writelines(str(cities) + '\r\n')
        f.writelines("RL:" + str(np.array(c_rl).mean())+ '\r\n')
        f.writelines("RL_LS_time:"+ str(np.mean(list(t_rl_ls)))+ '\r\n')
        f.writelines("RL_LS_cost:"+ str(np.mean(stop_costs[:,0]))+ '\r\n')
        f.writelines("LS1_time:"+ str(np.mean(list(t_ls)))+ '\r\n')
        f.writelines("LS1_cost:"+ str(np.mean(list(c_ls)))+ '\r\n')
        f.writelines("LS2_time:"+ str(np.mean(list(t_ls2)))+ '\r\n')
        f.writelines("LS2_cost:"+ str(np.mean(list(c_ls2)))+ '\r\n')

def test_batch_stop(test_LS=True, test_RL=True):

    if cities <= 50:
        model_path = '/home/kevin/PycharmProjects/CSP_Attention/outputs/csp_50/run_20200629T110102/epoch-31.pt'
    elif cities <= 150:
        model_path = '/home/kevin/PycharmProjects/CSP_Attention/outputs/csp_100/run_20200629T210739/epoch-30.pt'
    else:
        model_path = '/home/kevin/PycharmProjects/CSP_Attention/outputs/csp_200/run_20200630T170000/epoch-49.pt'
    model, _ = load_model(model_path)
    model.to(device)
    model.eval()

    test_nums = 50
    # 123: 7, 100city, LS(3.948),RL(4.19/4.06),ls-400
    # 123: 11, 100city, LS(3.118), RL(),ls-400
    # 123：7, 20citiy, LS(2.52), RL(2.15),ls,400
    # 123：7, 200citiy, LS(6.20), RL(6.78/6.98),ls,400
    # 123：7, 150citiy, LS(5.075), RL(6.78/5.12),ls,400
    seed_seed = 123456
    np.random.seed(seed_seed)
    seeds = np.random.randint(0, 123456, test_nums)
    print("Seeds:", seeds)

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

    stop_costs = np.array(list(c_rl_ls))

    print("RL_LS_cost:", np.mean(stop_costs[:,0]))



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

                stop_cost = stop_costs[np.where(stop_costs[:, 1] == loc[0,0])[0][0], 0]

                # worker
                pool.apply_async(LS_async, args=(loc, cover_range, c_ls, t_ls, c_ls2, t_ls2, stop_cost, ))

        pool.close()
        pool.join()
        print("LS1_time:", np.mean(list(t_ls)))
        print("LS1_cost:", np.mean(list(c_ls)))
        print("LS2_time:", np.mean(list(t_ls2)))
        print("LS2_cost:", np.mean(list(c_ls2)))

    if test_RL:
        print("RL:", np.array(c_rl).mean())
        print("RL_LS_time:", np.mean(list(t_rl_ls)))
        print("RL_LS_cost:", np.mean(stop_costs[:,0]))

    #     save
    with open(batch_stop_results_file_path, 'a+') as f:
        f.writelines(str(cities) + '\r\n')
        f.writelines("RL:" + str(np.array(c_rl).mean())+ '\r\n')
        f.writelines("RL_LS_time:"+ str(np.mean(list(t_rl_ls)))+ '\r\n')
        f.writelines("RL_LS_cost:"+ str(np.mean(stop_costs[:,0]))+ '\r\n')
        f.writelines("LS1_time:"+ str(np.mean(list(t_ls)))+ '\r\n')
        f.writelines("LS1_cost:"+ str(np.mean(list(c_ls)))+ '\r\n')
        f.writelines("LS2_time:"+ str(np.mean(list(t_ls2)))+ '\r\n')
        f.writelines("LS2_cost:"+ str(np.mean(list(c_ls2)))+ '\r\n')

def test_batch_radius(test_LS=True, radii = None, test_RL=True):

    if cities <= 50:
        model_path = '/home/kevin/PycharmProjects/CSP_Attention/outputs/csp_50/run_20200629T110102/epoch-31.pt'
    elif cities <= 150:
        model_path = '/home/kevin/PycharmProjects/CSP_Attention/outputs/csp_100/run_20200629T210739/epoch-30.pt'
    else:
        model_path = '/home/kevin/PycharmProjects/CSP_Attention/outputs/csp_200/run_20200630T170000/epoch-49.pt'
    model, _ = load_model(model_path)
    model.to(device)
    model.eval()

    test_nums = 50

    seed_seed = 123456
    np.random.seed(seed_seed)
    seeds = np.random.randint(0, 123456, test_nums)
    print("Seeds:", seeds)

    # radii = torch.ones(test_nums, cities)/5
    print(radii)
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
                                      radius=radii[i].unsqueeze(0),
                                      sample_mode=True)
            dataloader = DataLoader(dataset, batch_size=1)
            radius = radii[i].unsqueeze(0).cpu().numpy()
            for batch in dataloader:
                loc = batch['loc'].transpose(-1, -2)
                loc = loc[0, :, :].squeeze(0).transpose(0, 1).cpu().numpy()
                # worker
                pool.apply_async(LS_async_radius, args=(loc, cover_range, c_ls, t_ls, c_ls2, t_ls2, radius, 1e-6, ))

        pool.close()
        pool.join()
        print("LS1_time:", np.mean(list(t_ls)))
        print("LS1_cost:", np.mean(list(c_ls)))
        print("LS2_time:", np.mean(list(t_ls2)))
        print("LS2_cost:", np.mean(list(c_ls2)))


    c_rl = []
    c_rl_ls = []
    timee = 0
    # First obtain the RL results by sampling.
    for i in range(test_nums):

        dataset = Test_CSPDataset(size=cities, num_samples=num_sample,
                                  cover_range=cover_range, seed=seeds[i],
                                  radius=radii[i].unsqueeze(0),
                                  sample_mode=True)
        dataloader = DataLoader(dataset, batch_size=num_sample)
        model.set_decode_type("sampling")
        radius = radii[i].unsqueeze(0).cpu().numpy()
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
        t1 = time.time()
        opt_path, dists1, _ = LS(loc, tour=tour, cover_range=cover_range, render_if = False, tag='RL', radius = radius)
        timee = time.time()-t1
        c_rl_ls.append(dists1)
    print("RL:", np.array(c_rl).mean())
    print("RL_LS_time:", timee)
    print("RL_LS_cost:", np.mean(c_rl_ls))


    if test_LS:
        print("LS1_time:", np.mean(list(t_ls)))
        print("LS1_cost:", np.mean(list(c_ls)))
        print("LS2_time:", np.mean(list(t_ls2)))
        print("LS2_cost:", np.mean(list(c_ls2)))

    #     save
    with open(batch_radius_file_path, 'a+') as f:
        f.writelines(str(cities) + '\r\n')
        f.writelines("RL:" + str(np.array(c_rl).mean())+ '\r\n')
        f.writelines("RL_LS_time:"+ str(timee)+ '\r\n')
        f.writelines("RL_LS_cost:"+ str(np.mean(c_rl_ls))+ '\r\n')
        f.writelines("LS1_time:"+ str(np.mean(list(t_ls)))+ '\r\n')
        f.writelines("LS1_cost:"+ str(np.mean(list(c_ls)))+ '\r\n')
        f.writelines("LS2_time:"+ str(np.mean(list(t_ls2)))+ '\r\n')
        f.writelines("LS2_cost:"+ str(np.mean(list(c_ls2)))+ '\r\n')

def test_plot(cover_range, stop=False):
    if cities <= 50:
        model_path = '/home/kevin/PycharmProjects/CSP_Attention/outputs/csp_50/run_20200629T110102/epoch-31.pt'
    elif cities <= 150:
        model_path = '/home/kevin/PycharmProjects/CSP_Attention/outputs/csp_100/run_20200629T210739/epoch-30.pt'
        # model_path = '/home/kevin/PycharmProjects/CSP_Attention/outputs/csp_100/run_20201103T115209/epoch-30.pt'
    else:
        model_path = '/home/kevin/PycharmProjects/CSP_Attention/outputs/csp_200/run_20200630T170000/epoch-49.pt'

    model, _ = load_model(model_path)
    model.to(device)
    model.eval()

    # # Test of instances with cover radius
    # dataset = Test_CSPDataset(size=cities, num_samples=num_sample,
    #                           cover_range=cover_range, seed=1234, radius=torch.ones(1,cities)/5,
    #                           sample_mode=True)
    dataset = Test_CSPDataset(size=cities, num_samples=num_sample,
                              cover_range=cover_range, seed=123456, radius=torch.rand(1,cities)/5,
                              sample_mode=True)
    # Test of instances with cover number (NC)
    # dataset = Test_CSPDataset(size=cities, num_samples=num_sample,
    #                           cover_range=cover_range, seed=1234, radius=None,
    #                           sample_mode=True)

    # seed = 1234 produces good figures
    # # Test of instances with variable cover number (NC)
    # dataset = Test_CSPDataset(size=cities, num_samples=num_sample,
    #                           cover_range=cover_range, seed=1234, radius=None,
    #                           sample_mode=True, varible_NC=True)

    if not isinstance(dataset[0]['cover_range'], int):
        cover_range = dataset[0]['cover_range'].numpy()

    dataloader = DataLoader(dataset, batch_size=num_sample)

    radius = None
    if 'radius' in dataset[0].keys():
        radius = dataset[0]['radius']
        if isinstance(radius, torch.Tensor):
            radius = radius.cpu().numpy()

    model.set_decode_type("sampling")
    costs = np.ones((3,))
    for batch in dataloader:
        batch = move_to(batch, device)
        with torch.no_grad():
            t1 = time.time()
            cost, logp, tour = model(batch, return_pi=True)
            dists, len = tour_len_dist(batch, tour)
            # print(time.time() - t1)
            sorted, inds = dists.sort()
            print("RL:", dists.min())
            print(time.time()-t1)
            loc = batch['loc'].transpose(-1, -2)

            # render(loc[inds[:9]], tour[inds[:9]], os.path.join('plot_dir', 'test_%2.3f.png' % (cost.mean().item())))

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

            opt_path, dists1, _ = LS(loc, tour=tour, cover_range=cover_range, tag='RL', radius = radius)
            print("RL, afer_ls:", dist(loc, tour), dist(loc, opt_path))
            costs[0]=dists1
    if not stop:
        final_tour, dists_LS, _ = LS(loc, tour=None, cover_range=cover_range, tag='LS1', radius = radius)
    else:
        final_tour, dists_LS, _ = LS(loc, tour=None, cover_range=cover_range, tag='LS1', stop_cost=dist(loc, opt_path), radius = radius)
    costs[1]=dists_LS
    print("LS1:", dists_LS)
    # if not stop:
    #     final_tour, dists_LS2, _ = LS(loc, tour=None, cover_range=cover_range, tag='LS2')
    # else:
    #     final_tour, dists_LS2, _ = LS(loc, tour=None, cover_range=cover_range, tag='LS2', stop_cost=dist(loc, opt_path))
    # costs[2]=dists_LS2
    # print("LS2:", dists_LS2)
    return costs

if __name__ == '__main__':
    cover_range = 7
    num_sample = 1000
    cities = 100
    test_plot_if = True

    if test_plot_if:
        # test single instance
        test_plot(cover_range)
        # test nbatch
        # all_costs = []
        # test_nums = 20
        # cities_ = [50, 100, 150, 200]
        #
        # for cities in cities_:
        #     costs = np.zeros((test_nums, 3))
        #     for i in range(test_nums):
        #         costs[i] = test_plot(cover_range)
        #     all_costs.append(costs.mean(0))
        # print(all_costs)
    # [2.54119728, 2.30990402, 2.36474072]), array([3.01252633, 2.96680852, 3.02643683]), array([3.70551018, 3.65829661, 3.77317828]), array([4.20810802, 4.14878808, 4.27242118])
    else:
        # cities_ = [50, 100, 150, 200]
        cities_ = [300]
        batch_results_file_path = '../compare_results/batch_test_11.txt'
        batch_stop_results_file_path = '../compare_results/batch_stop_test_11.txt'
        batch_radius_file_path = '../compare_results/batch_radius_all.txt'
        import time
        if not os.path.exists(batch_results_file_path):
            os.system(r"touch {}".format(batch_results_file_path))#调用系统命令行来创建文件
        if not os.path.exists(batch_radius_file_path):
            os.system(r"touch {}".format(batch_results_file_path))#调用系统命令行来创建文件
        if not os.path.exists(batch_stop_results_file_path):
            os.system(r"touch {}".format(batch_stop_results_file_path))#调用系统命令行来创建文件

        with open(batch_results_file_path, 'a+') as f:
            f.writelines(time.strftime("%Y-%m-%d-%H:%M:%S") + '\r\n')
        with open(batch_stop_results_file_path, 'a+') as f:
            f.writelines(time.strftime("%Y-%m-%d-%H:%M:%S") + '\r\n')
        with open(batch_radius_file_path, 'a+') as f:
            f.writelines(time.strftime("%Y-%m-%d-%H:%M:%S") + '\r\n')
        for i in range(len(cities_)):
            cities = cities_[i]
            if cities == 300:
                num_sample = 1000
            elif cities == 200:
                num_sample = 1500
            else:
                num_sample = 3000
            test_batch()
        #     test_batch_radius()
        # radii_varible = torch.rand(len(cities_), 50, 300)/4
        # for i in range(len(cities_)):
        #     cities = cities_[i]
        #     if cities == 300:
        #         num_sample = 1000
        #     elif cities == 200:
        #         num_sample = 1500
        #     else:
        #         num_sample = 3000
        #     test_batch_radius(radii=radii_varible[i, :, :cities])
        #     test_batch_radius(radii=torch.ones(50, cities)/5)
        # test_batch_stop()
        # test_active_search()


