from utils.functions import *
from test_plot.test_dataset import Test_CSPDataset, render, tour_len_dist
from torch.utils.data import DataLoader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_path = "/home/kevin/PycharmProjects/attention-learn-to-route/outputs/csp_50/run_20191107T175019/epoch-20.pt"
model, _ = load_model(model_path)

dataset = Test_CSPDataset(size=400, num_samples=10, cover_range=11, seed=13)
# dataset = Test_CSPDataset(size=52, num_samples=1, seed=13, tsp_name='berlin52')
# dataset = Test_CSPDataset(size=60, num_samples=1, seed=13, tsp_name='kroA100')

dataloader = DataLoader(dataset, batch_size=10)

model.to(device)
model.eval()
model.set_decode_type("greedy")

import time
for batch in dataloader:
    batch = move_to(batch, device)
    with torch.no_grad():
        t1=time.time()
        cost, logp, tour = model(batch, return_pi=True)
        print(time.time()-t1)
        print(tour_len_dist(batch, tour))

        batch = batch['loc'].transpose(-1,-2)
        render(batch,tour,os.path.join('plot_dir', 'test_%2.3f.png'%(cost.mean().item())))

