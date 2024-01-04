import numpy
import torch
import torch.nn.functional as F
import numpy as np
from FLAlgorithms.users.userbase import User

class userFedHKD(User):
    def __init__(self,
                 args, id, model,
                 train_data, test_data,
                 available_labels, latent_layer_idx, label_info, output_dim, latent_dim,
                 use_adam=False):
        super().__init__(args, id, model, train_data, test_data, use_adam=use_adam)
        self.gen_batch_size = args.gen_batch_size
        self.latent_layer_idx = latent_layer_idx
        self.available_labels = available_labels
        self.label_info=label_info
        self.local_prototype = torch.zeros((output_dim, latent_dim))
        self.local_softlabel = torch.zeros((output_dim, output_dim))


    def exp_lr_scheduler(self, epoch, decay=0.98, init_lr=0.1, lr_decay_epoch=1):
        """Decay learning rate by a factor of 0.95 every lr_decay_epoch epochs."""
        lr= max(1e-4, init_lr * (decay ** (epoch // lr_decay_epoch)))
        return lr

    def exp_down_lr_scheduler(self, epoch, decay=0.98, init_lr=0.1, lr_decay_epoch=1, down_lr=1e-5):
        """
        Decay learning rate by a factor of `decay` every `lr_decay_epoch` epochs,
        with a lower bound of `down_lr` for the learning rate.
        """
        lr = max(down_lr, init_lr * (decay ** (epoch // lr_decay_epoch)))
        return lr

    def update_label_counts(self, labels, counts):
        for label, count in zip(labels, counts):
            self.label_counts[int(label)] += count

    def clean_up_counts(self):
        del self.label_counts
        self.label_counts = {label:1 for label in range(self.unique_labels)}

    def train(self, glob_iter,glob_knowledge,personalized=False, early_stop=100, regularization=True, verbose=False):
        self.clean_up_counts()
        self.local_prototype.fill_(0)
        self.local_softlabel.fill_(0)
        self.model.train()
        Know_LOSS, Pre_LOSS, Prot_LOSS = 0, 0, 0#
        for epoch in range(self.local_epochs):
            self.optimizer.zero_grad()
            #### sample from real dataset (un-weighted)
            samples =self.get_next_train_batch(count_labels=True)
            X, y = samples['X'], samples['y']
            self.update_label_counts(samples['labels'], samples['counts'])
            model_result= self.model(X, logit=True)
            user_output_logp = model_result['output']
            user_logit_logp = model_result['logit']
            T = 0.5
            temp_softlabel = F.softmax(user_logit_logp / T, dim=1)
            temp_prot = model_result['samples_prototype']
            for i in range(len(y)):
                self.local_softlabel[y[i]] += temp_softlabel[i,:].clone().detach()
                self.local_prototype[y[i]] += temp_prot[i,:].clone().detach()

            ######## Predicte loss ########
            predictive_loss=self.loss(user_output_logp, y)

            #### sample y and generate z
            if regularization and epoch < early_stop:
                generative_alpha=0.05
                generative_beta =0.05
                ######## Prot_LOSS loss ########
                prot_loss = self.dist_loss(temp_prot, glob_knowledge[0][y])

                ######## Knowledge_loss#########
                know_logit =self.model(glob_knowledge[0], start_layer_idx= -1 ,logit = True)['logit']
                know_output_T = F.softmax(know_logit / T, dim=1)
                Knowledge_loss = self.dist_loss(know_output_T, glob_knowledge[1])
                loss = predictive_loss + generative_alpha * prot_loss + generative_beta * Knowledge_loss
                Pre_LOSS+=predictive_loss
                Prot_LOSS+=prot_loss
                Know_LOSS+=Knowledge_loss

            else:
                #### get loss and perform optimization
                loss=predictive_loss
            loss.backward()
            self.optimizer.step()#self.local_model)

        self.clone_model_paramenter(self.model.parameters(), self.local_model)
        if personalized:
            self.clone_model_paramenter(self.model.parameters(), self.personalized_model_bar)
        self.lr_scheduler.step(glob_iter)
        for key, value in self.label_counts.items(): #aggregate feature
            if value - 1 != 0:
                self.local_prototype[key] /= (value - 1)
                self.local_softlabel[key] /= (value - 1)

        noise_gauss = np.random.normal(0, 0.0200, self.local_prototype.shape)
        self.local_prototype +=noise_gauss

        if regularization and verbose:
            Know_LOSS = Know_LOSS.detach().numpy() / self.local_epochs
            Pre_LOSS = Pre_LOSS.detach().numpy() / self.local_epochs
            Prot_LOSS = Prot_LOSS.detach().numpy() / self.local_epochs
            info='\nUser Know Loss={:.4f}'.format(Know_LOSS)
            info+=', Pre Loss={:.4f}'.format(Pre_LOSS)
            info += ', Prot Loss={:.4f}'.format(Prot_LOSS)
            print(info)

    def adjust_weights(self, samples):
        labels, counts = samples['labels'], samples['counts']
        #weight=self.label_weights[y][:, user_idx].reshape(-1, 1)
        np_y = samples['y'].detach().numpy()
        n_labels = samples['y'].shape[0]
        weights = np.array([n_labels / count for count in counts]) # smaller count --> larger weight
        weights = len(self.available_labels) * weights / np.sum(weights) # normalized
        label_weights = np.ones(self.unique_labels)
        label_weights[labels] = weights
        sample_weights = label_weights[np_y]
        return sample_weights


