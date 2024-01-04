import numpy
import torch
import torch.nn.functional as F
import numpy as np
from FLAlgorithms.users.userbase import User

class UserFedPA(User):
    def __init__(self,
                 args, id, model, generative_model,
                 train_data, test_data,
                 available_labels, latent_layer_idx, label_info, output_dim, latent_dim,
                 use_adam=False):
        super().__init__(args, id, model, train_data, test_data, use_adam=use_adam)
        self.gen_batch_size = args.gen_batch_size
        self.generative_model = generative_model
        self.latent_layer_idx = latent_layer_idx
        self.available_labels = available_labels
        self.label_info=label_info
        self.local_prototype = torch.zeros((output_dim, latent_dim))


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

    def train(self, glob_iter,glob_prototype,personalized=False, early_stop=100, regularization=True, verbose=False):
        self.clean_up_counts()
        self.local_prototype.fill_(0)
        self.model.train()
        self.generative_model.eval()
        GenSam_LOSS, Pre_LOSS, Prot_LOSS = 0, 0, 0#
        for epoch in range(self.local_epochs):
            self.optimizer.zero_grad()
            #### sample from real dataset (un-weighted)
            samples =self.get_next_train_batch(count_labels=True)
            X, y = samples['X'], samples['y']
            self.update_label_counts(samples['labels'], samples['counts'])
            model_result= self.model(X, logit=True)
            user_output_logp = model_result['output']
            temp_prot = model_result['samples_prototype']
            for i in range(len(y)):
                self.local_prototype[y[i]] += temp_prot[i,:].clone().detach()
            ######## Predicte loss ########
            predictive_loss=self.loss(user_output_logp, y)

            #### sample y and generate z
            if regularization and epoch < early_stop:
                generative_alpha=self.exp_lr_scheduler(glob_iter, decay=0.98, init_lr=25)

                generative_beta=self.exp_down_lr_scheduler(glob_iter, decay=0.98, init_lr=5,down_lr=0.15)
                sampled_y=np.random.choice(self.available_labels, self.gen_batch_size)
                sampled_y=torch.tensor(sampled_y)
                gen_result=self.generative_model(sampled_y, latent_layer_idx=self.latent_layer_idx)
                gen_output=gen_result['output'] # latent representation when latent = True, x otherwise

                ######## GenSam loss ########
                user_output_logp =self.model(gen_output, start_layer_idx=self.latent_layer_idx)['output']
                gensam_loss =  generative_alpha * torch.mean(#gen_loss
                    self.loss(user_output_logp, sampled_y)
                )

                ######## Prot_LOSS loss ########
                prot_loss = generative_beta * self.dist_loss(temp_prot, glob_prototype[y])

                # this is to further balance oversampled down-sampled synthetic data
                gen_ratio = self.gen_batch_size / self.batch_size
                loss = predictive_loss + gen_ratio * gensam_loss + prot_loss

                GenSam_LOSS+=gensam_loss
                Pre_LOSS+=predictive_loss
                Prot_LOSS+=prot_loss

            else:
                #### get loss and perform optimization
                loss=predictive_loss
            loss.backward()
            self.optimizer.step()#self.local_model)
        # local-model <=== self.model
        self.clone_model_paramenter(self.model.parameters(), self.local_model)
        if personalized:
            self.clone_model_paramenter(self.model.parameters(), self.personalized_model_bar)
        self.lr_scheduler.step(glob_iter)
        for key, value in self.label_counts.items(): #aggregate feature
            if value - 1 != 0:
                self.local_prototype[key] /= (value - 1)

        if regularization and verbose:
            GenSam_LOSS=GenSam_LOSS.detach().numpy() / self.local_epochs
            Pre_LOSS = Pre_LOSS.detach().numpy() / self.local_epochs
            Prot_LOSS = Prot_LOSS.detach().numpy() / self.local_epochs
            info='\nUser GenSample Loss={:.4f}'.format(GenSam_LOSS)
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


