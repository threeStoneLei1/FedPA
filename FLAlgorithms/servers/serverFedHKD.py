import numpy

from FLAlgorithms.users.userFedHKD import userFedHKD
from FLAlgorithms.servers.serverbase import Server
from utils.model_utils import read_data, read_user_data, aggregate_user_data, create_generative_model
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.utils import save_image
import os
import copy
import time
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

MIN_SAMPLES_PER_LABEL = 1


class FedHKD(Server):
    def __init__(self, args, model, seed):
        super().__init__(args, model, seed)

        # Initialize data for all users
        data = read_data(args.dataset)
        # data contains: clients, groups, train_data, test_data, proxy_data
        clients = data[0]
        total_users = len(clients)
        self.total_test_samples = 0
        self.local = 'local' in self.algorithm.lower()
        self.use_adam = 'adam' in self.algorithm.lower()

        self.early_stop = 20  # stop using generated samples after 20 local epochs
        self.student_model = copy.deepcopy(self.model)

        self.glob_knowledge = [torch.zeros((self.model.output_dim, self.model.latent_dim)),
                               torch.zeros((self.model.output_dim, self.model.output_dim))]

        if not args.train:
            print('number of model parameteres: [{}]'.format(self.model.get_number_of_parameters()))
        self.latent_layer_idx = -1
        self.init_ensemble_configs()
        print("latent_layer_idx: {}".format(self.latent_layer_idx))
        print("ensemeble learning rate: {}".format(self.ensemble_lr))
        print("ensemeble alpha = {}, beta = {}, eta = {}".format(self.ensemble_alpha, self.ensemble_beta,
                                                                 self.ensemble_eta))
        print("generator alpha = {}, beta = {}".format(self.generative_alpha, self.generative_beta))
        self.init_loss_fn()
        self.train_data_loader, self.train_iter, self.available_labels = aggregate_user_data(data, args.dataset,
                                                                                             self.ensemble_batch_size)
        self.optimizer = torch.optim.Adam(
            params=self.model.parameters(),
            lr=self.ensemble_lr, betas=(0.9, 0.999),
            eps=1e-08, weight_decay=0, amsgrad=False)
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=0.98)

        #### creating users ####
        self.users = []
        for i in range(total_users):
            id, train_data, test_data, label_info = read_user_data(i, data, dataset=args.dataset, count_labels=True)
            self.total_train_samples += len(train_data)
            self.total_test_samples += len(test_data)
            id, train, test = read_user_data(i, data, dataset=args.dataset)
            user = userFedHKD(
                args, id, model,
                train_data, test_data,
                self.available_labels, self.latent_layer_idx, label_info, self.model.output_dim, self.model.latent_dim,
                use_adam=self.use_adam)
            self.users.append(user)
        print("Number of Train/Test samples:", self.total_train_samples, self.total_test_samples)
        print("Data from {} users in total.".format(total_users))
        print("Finished creating FedPA server.")

    def train(self, args):
        #### pretraining
        for glob_iter in range(self.num_glob_iters):
            print("\n\n-------------Round number: ", glob_iter, " -------------\n\n")
            self.selected_users, self.user_idxs = self.select_users(glob_iter, self.num_users, return_idx=True)
            if not self.local:
                self.send_parameters(mode=self.mode)  # broadcast averaged prediction model

            self.evaluate()
            chosen_verbose_user = np.random.choice(self.user_idxs)
            self.timestamp = time.time()  # log user-training start time
            array_prototype = []
            array_softlabel = []
            array_labelinfo = []
            threhold = 0.25
            for user_id, user in zip(self.user_idxs, self.selected_users):  # allow selected users to train
                verbose = user_id == chosen_verbose_user
                # perform regularization using generated samples after the first communication round
                user.train(
                    glob_iter,
                    personalized=self.personalized,
                    early_stop=self.early_stop,
                    glob_knowledge=self.glob_knowledge,
                    verbose=verbose and glob_iter > 0,
                    regularization=glob_iter > 0,
                )
                array_prototype.append(user.local_prototype)
                array_softlabel.append(user.local_softlabel)
                array_labelinfo.append(user.label_info)
            curr_timestamp = time.time()  # log  user-training end time
            train_time = (curr_timestamp - self.timestamp) / len(self.selected_users)
            self.metrics['user_train_time'].append(train_time)
            if self.personalized:
                self.evaluate_personalized_model()
            self.timestamp = time.time()  # log server-agg start tim
            self.aggregate_parameters()
            temp_prototype, temp_softlabel = self.calculate_knowledge(array_prototype, array_softlabel, array_labelinfo,
                                                                      threhold)
            self.glob_knowledge[0] = temp_prototype
            self.glob_knowledge[1] = temp_softlabel
            curr_timestamp = time.time()  # log  server-agg end time
            agg_time = curr_timestamp - self.timestamp
            self.metrics['server_agg_time'].append(agg_time)

        self.save_results(args)
        self.save_model()

    def get_label_weights(self):
        label_weights = []
        qualified_labels = []
        for label in range(self.unique_labels):
            weights = []
            for user in self.selected_users:
                weights.append(user.label_counts[label])
            if np.max(weights) > MIN_SAMPLES_PER_LABEL:
                qualified_labels.append(label)
            # uniform
            label_weights.append(np.array(weights) / np.sum(weights))
        label_weights = np.array(label_weights).reshape((self.unique_labels, -1))
        return label_weights, qualified_labels

    def calculate_knowledge(self, array_prototype, array_softlabel, array_labelinfo, threhold):
        temp_prototype = torch.zeros(self.glob_knowledge[0].shape)
        temp_softlabel = torch.zeros(self.glob_knowledge[1].shape)
        count_sum_useri = np.zeros(len(array_labelinfo))
        selected_user_labelinfo = {}
        for index, useri_labinfo in enumerate(array_labelinfo):
            for labels, counts in zip(useri_labinfo['labels'], useri_labinfo['counts']):
                count_sum_useri[index] += counts
                if labels in selected_user_labelinfo:
                    selected_user_labelinfo[labels] += counts
                else:
                    selected_user_labelinfo[labels] = counts

        for index, (useri_prot, useri_soft, useri_labinfo) in enumerate(
                zip(array_prototype, array_softlabel, array_labelinfo)):
            for labels, counts in zip(useri_labinfo['labels'], useri_labinfo['counts']):
                if counts / count_sum_useri[index] < threhold:
                    selected_user_labelinfo[labels] -= counts

        for index, (useri_prot, useri_soft, useri_labinfo) in enumerate(
                zip(array_prototype, array_softlabel, array_labelinfo)):
            for labels, counts in zip(useri_labinfo['labels'], useri_labinfo['counts']):
                if counts / count_sum_useri[index] >= threhold:
                    aggre_alpha = (
                        counts / selected_user_labelinfo[labels] if not selected_user_labelinfo[labels] == 0 else 0)
                    temp_prototype[labels] += aggre_alpha * useri_prot[labels]
                    temp_softlabel[labels] += aggre_alpha * useri_soft[labels]

        return temp_prototype, temp_softlabel


