import numpy
from FLAlgorithms.users.userFedPA import UserFedPA
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


class FedPA(Server):
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
        self.generative_model = create_generative_model(args.dataset, args.algorithm, self.model_name, args.embedding)

        self.glob_prototype = torch.zeros((self.model.output_dim, self.model.latent_dim))  # global prototype

        if not args.train:
            print('number of generator parameteres: [{}]'.format(self.generative_model.get_number_of_parameters()))
            print('number of model parameteres: [{}]'.format(self.model.get_number_of_parameters()))
        self.latent_layer_idx = self.generative_model.latent_layer_idx
        self.init_ensemble_configs()
        print("latent_layer_idx: {}".format(self.latent_layer_idx))
        print("label embedding {}".format(self.generative_model.embedding))
        print("ensemeble learning rate: {}".format(self.ensemble_lr))
        print("ensemeble alpha = {}, beta = {}, eta = {}".format(self.ensemble_alpha, self.ensemble_beta,
                                                                 self.ensemble_eta))
        print("generator alpha = {}, beta = {}".format(self.generative_alpha, self.generative_beta))
        self.init_loss_fn()
        self.train_data_loader, self.train_iter, self.available_labels = aggregate_user_data(data, args.dataset,
                                                                                             self.ensemble_batch_size)
        self.generative_optimizer = torch.optim.Adam(
            params=self.generative_model.parameters(),
            lr=self.ensemble_lr, betas=(0.9, 0.999),
            eps=1e-08, weight_decay=self.weight_decay, amsgrad=False)
        self.generative_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.generative_optimizer, gamma=0.98)
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
            user = UserFedPA(
                args, id, model, self.generative_model,
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
            array_labelinfo = []
            for user_id, user in zip(self.user_idxs, self.selected_users):  # allow selected users to train
                verbose = user_id == chosen_verbose_user
                # perform regularization using generated samples after the first communication round
                user.train(
                    glob_iter,
                    personalized=self.personalized,
                    early_stop=self.early_stop,
                    glob_prototype=self.glob_prototype,
                    verbose=verbose and glob_iter > 0,
                    regularization=glob_iter > 0
                )
                array_prototype.append(user.local_prototype)
                array_labelinfo.append(user.label_info)
            curr_timestamp = time.time()  # log  user-training end time
            train_time = (curr_timestamp - self.timestamp) / len(self.selected_users)
            self.metrics['user_train_time'].append(train_time)
            if self.personalized:
                self.evaluate_personalized_model()
            self.timestamp = time.time()  # log server-agg start time
            self.train_generator(
                self.batch_size,
                epoches=self.ensemble_epochs // self.n_teacher_iters,
                latent_layer_idx=self.latent_layer_idx,
                glob_prototype=self.glob_prototype,
                glob_iter=glob_iter,
                verbose=True
            )
            self.aggregate_parameters()
            self.glob_prototype = self.calculate_prototype(array_prototype, array_labelinfo)
            curr_timestamp = time.time()  # log  server-agg end time
            agg_time = curr_timestamp - self.timestamp
            self.metrics['server_agg_time'].append(agg_time)
            if glob_iter > 0 and glob_iter % 20 == 0 and self.latent_layer_idx == 0:
                self.visualize_images(self.generative_model, glob_iter, repeats=10)

        self.save_results(args)
        self.save_model()

    def train_generator(self, batch_size, glob_prototype, glob_iter, epoches=1, latent_layer_idx=-1, verbose=False):

        # self.generative_regularizer.train()
        self.label_weights, self.qualified_labels = self.get_label_weights()
        Adversarial_LOSS, Prototype_LOSS, Diversity_LOSS = 0, 0, 0

        def update_generator_(n_iters, student_model, glob_prototype, glob_iter, Adversarial_LOSS, Prototype_LOSS,
                              Diversity_LOSS):
            self.generative_model.train()
            student_model.eval()
            for i in range(n_iters):
                self.generative_optimizer.zero_grad()
                y = np.random.choice(self.qualified_labels, batch_size)
                y_input = torch.LongTensor(y)
                ## feed to generator
                gen_result = self.generative_model(y_input, latent_layer_idx=latent_layer_idx, verbose=True)
                # get approximation of Z( latent) if latent set to True, X( raw image) otherwise
                gen_output, eps = gen_result['output'], gen_result['eps']

                ######### get Diversity loss ############
                isPA = True
                diversity_loss = self.generative_model.diversity_loss(eps, gen_output, y_input,
                                                                      isPA)  # encourage different outputs

                ######### get Adversarial loss ############
                advers_loss = 0
                for user_idx, user in enumerate(self.selected_users):
                    user.model.eval()
                    weight = self.label_weights[y][:, user_idx].reshape(-1, 1)
                    # expand_weight=np.tile(weight, (1, self.unique_labels))
                    user_result_given_gen = user.model(gen_output, start_layer_idx=latent_layer_idx, logit=True)
                    user_output_logp_ = F.log_softmax(user_result_given_gen['logit'], dim=1)
                    advers_loss_ = torch.mean( \
                        self.generative_model.crossentropy_loss(user_output_logp_, y_input) * \
                        torch.tensor(weight, dtype=torch.float32))
                    advers_loss += advers_loss_  # 用每一个客户端分类器训练生成器 保真  L_re

                ensemble_alpha = user.exp_lr_scheduler(glob_iter, decay=0.98, init_lr=25)
                ensemble_beta = 0.15
                ######### get Prototype loss ############
                prototype_loss = torch.zeros_like(advers_loss)
                if glob_iter > 0:

                    prototype_loss = ensemble_beta * self.generative_model.dist_loss(gen_output, glob_prototype[y])

                    loss = ensemble_alpha * advers_loss + (-1 * prototype_loss) + diversity_loss
                else:
                    loss = ensemble_alpha * advers_loss + diversity_loss
                loss.backward()
                self.generative_optimizer.step()
                Adversarial_LOSS += ensemble_alpha * advers_loss  # (torch.mean(TEACHER_LOSS.double())).item()
                Prototype_LOSS += ensemble_beta * prototype_loss
                Diversity_LOSS += self.ensemble_eta * diversity_loss  # (torch.mean(diversity_loss.double())).item()
            return Adversarial_LOSS, Prototype_LOSS, Diversity_LOSS

        for i in range(epoches):
            Adversarial_LOSS, Prototype_LOSS, Diversity_LOSS = update_generator_(
                self.n_teacher_iters, self.model, glob_prototype, glob_iter, Adversarial_LOSS, Prototype_LOSS,
                Diversity_LOSS)

        Adversarial_LOSS = Adversarial_LOSS.detach().numpy() / (self.n_teacher_iters * epoches)
        Prototype_LOSS = Prototype_LOSS.detach().numpy() / (self.n_teacher_iters * epoches)
        Diversity_LOSS = Diversity_LOSS.detach().numpy() / (self.n_teacher_iters * epoches)
        info = "Generator: Adversarial_LOSS= {:.4f}, Prototype_LOSS= {:.4f}, Diversity_LOSS = {:.4f}, ". \
            format(Adversarial_LOSS, Prototype_LOSS, Diversity_LOSS)
        if verbose:
            print(info)
        self.generative_lr_scheduler.step()

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

    def visualize_images(self, generator, glob_iter, repeats=1):
        """
        Generate and visualize data for a generator.
        """
        os.system("mkdir -p images")
        path = f'images/{self.algorithm}-{self.dataset}-iter{glob_iter}.png'
        y = self.available_labels
        y = np.repeat(y, repeats=repeats, axis=0)
        y_input = torch.tensor(y)
        generator.eval()
        images = generator(y_input, latent=False)['output']  # 0,1,..,K, 0,1,...,K
        images = images.view(repeats, -1, *images.shape[1:])
        images = images.view(-1, *images.shape[2:])
        save_image(images.detach(), path, nrow=repeats, normalize=True)
        print("Image saved to {}".format(path))

    def calculate_prototype(self, array_prototype, array_labelinfo):
        temp_prototype = torch.zeros(self.glob_prototype.shape)
        selected_user_labelinfo = {}
        for useri_labinfo in array_labelinfo:
            for labels, counts in zip(useri_labinfo['labels'], useri_labinfo['counts']):
                if labels in selected_user_labelinfo:
                    selected_user_labelinfo[labels] += counts
                else:
                    selected_user_labelinfo[labels] = counts
        for useri_prot, useri_labinfo in zip(array_prototype, array_labelinfo):
            for labels, counts in zip(useri_labinfo['labels'], useri_labinfo['counts']):
                aggre_alpha = (
                    counts / selected_user_labelinfo[labels] if not selected_user_labelinfo[labels] == 0 else 0)
                temp_prototype[labels] += aggre_alpha * useri_prot[labels]

        return temp_prototype

    def visualize_iid_data(self):
        user = self.users
        num_users = len(user)

        # 找出所有存在的标签
        all_labels = set()
        for u in user:
            all_labels.update(u.label_info['labels'])

        num_labels = len(all_labels)
        max_count = 0
        min_count = 0

        label_counts = np.zeros((num_users, num_labels))
        for user_id, u in enumerate(user):
            labels = u.label_info['labels']
            counts = u.label_info['counts']
            for label, count in zip(labels, counts):
                label_counts[user_id][label] = count
                if count > max_count:
                    max_count = count

        norm = mcolors.Normalize(vmin=min_count, vmax=max_count)
        cmap = plt.cm.get_cmap('Purples')


        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(label_counts, cmap=cmap, aspect='auto', norm=norm)


        ax.set_xticks(np.arange(num_labels))
        ax.set_xticklabels(sorted(all_labels))
        ax.set_yticks(np.arange(num_users))
        ax.set_yticklabels(np.arange(1, num_users + 1))
        ax.set_xlabel('label')
        ax.set_ylabel('user ID')


        cbar = plt.colorbar(im)
        cbar.set_label('label counts')


        plt.subplots_adjust(bottom=0.2)

        alpha_index = self.dataset.find("alpha")
        alpha_value = self.dataset[alpha_index + len("alpha"):]
        alpha_value = alpha_value.split("-")[0]
        # alpha_value = self.dataset.split('-ratio')[0].split()
        # plt.annotate(r'$\alpha = {}$'.format(alpha_value), xy=(0.5, -0.2), xycoords='axes fraction', ha='center', fontsize=15)
        fig_name = "/code/FedPA/iid_figs/" + self.dataset.split('-ratio')[0] + ".png"
        plt.savefig(fig_name, bbox_inches='tight')

        plt.show()