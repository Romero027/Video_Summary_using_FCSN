import torch
import torch.optim as optim

from tensorboardX import SummaryWriter
import numpy as np
import json
import os
from tqdm import tqdm, trange
import h5py
from prettytable import PrettyTable

from fcsn import FCSN
import eval
import time


class Solver(object):
    """Class that Builds, Trains FCSN model"""

    def __init__(self, config=None, train_loader=None, test_dataset=None):
        self.config = config
        self.train_loader = train_loader
        self.test_dataset = test_dataset

        # model
        self.model = FCSN(self.config.n_class)

        # optimizer
        if self.config.mode == 'train':
            self.optimizer = optim.Adam(self.model.parameters())
            # self.optimizer = optim.SGD(self.model.parameters(), lr=config.lr, momentum=self.config.momentum)
            self.model.train()

        if self.config.gpu:
            self.model = self.model.cuda()

        if not os.path.exists(self.config.score_dir):
            os.mkdir(self.config.score_dir)

        if not os.path.exists(self.config.save_dir):
            os.mkdir(self.config.save_dir)

        if not os.path.exists(self.config.log_dir):
            os.mkdir(self.config.log_dir)

    @staticmethod
    def sum_loss(pred_score, gt_labels, weight=None):
        n_batch, n_class, n_frame = pred_score.shape
        log_p = torch.log_softmax(pred_score, dim=1).reshape(-1, n_class)
        gt_labels = gt_labels.reshape(-1)
        criterion = torch.nn.NLLLoss(weight)
        loss = criterion(log_p, gt_labels)
        return loss


    def evaluate(self, model_path):
        self.model.eval()
        out_dict = {}
        eval_arr = []
        table = PrettyTable()
        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint)
        table.title = 'Eval result'
        table.field_names = ['ID', 'Precision', 'Recall', 'F-score']
        table.float_format = '1.3'
        
        inference_time = []
        with h5py.File(self.config.data_path) as data_file:
            for feature, label, idx in tqdm(self.test_dataset, desc='Evaluate', ncols=80, leave=False):
                if self.config.gpu:
                    feature = feature.cuda()


                start = time.time()
                print(feature.size())
                pred_score = self.model(feature.unsqueeze(0)).squeeze(0)
                inference_time.append(time.time() - start)


                pred_score = torch.softmax(pred_score, dim=0)[1]
                video_info = data_file['video_'+str(idx)]
                pred_score, pred_selected, pred_summary = eval.select_keyshots(video_info, pred_score)
                true_summary_arr = video_info['user_summary'][()]
                print(len(pred_summary), len(true_summary_arr[0]))
                eval_res = [eval.eval_metrics(pred_summary, true_summary) for true_summary in true_summary_arr]
                eval_res = np.mean(eval_res, axis=0).tolist()

                eval_arr.append(eval_res)
                table.add_row([idx] + eval_res)

                out_dict[idx] = {
                    'pred_score': pred_score, 
                    'pred_selected': pred_selected, 'pred_summary': pred_summary
                    }
                    
        eval_mean = np.mean(eval_arr, axis=0).tolist()
        table.add_row(['mean']+eval_mean)
        tqdm.write(str(table))
        print(inference_time)


if __name__ == '__main__':
    from config import Config
    from data_loader import get_loader
    train_config = Config()
    test_config = Config(mode='test')
    train_loader, test_dataset = get_loader(train_config.data_path, batch_size=train_config.batch_size)
    solver = Solver(train_config, train_loader, test_dataset)
    model_path = "/home/ubuntu/Video_Summary_using_FCSN/save_dir/epoch-99.pkl"
    solver.evaluate(model_path)
