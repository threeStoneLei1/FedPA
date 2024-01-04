from FLAlgorithms.users.userpFedGen import UserpFedGen
from FLAlgorithms.servers.serverbase import Server
from utils.model_utils import read_data, read_user_data, aggregate_user_data, create_generative_model
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.utils import save_image
import os
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import math
import seaborn as sns

class visual_gene_data_TSNE:
    def __init__(self,generator, glob_iter, algorithm, dataset, available_labels, repeats=150):
        self.generator = generator
        self.glob_iter = glob_iter
        self.repeats = repeats
        self.algorithm = algorithm
        self.dataset = dataset
        self.available_labels = available_labels


    def visualize_images(self):
        """
        Generate and visualize data for a generator.
        """
        generator = self.generator
        os.system(f"mkdir -p images")
        path = f'images/{self.algorithm}-{self.dataset}-iter{self.glob_iter}.png'
        y = self.available_labels
        y_label = np.repeat(y, repeats=self.repeats, axis=0)
        y_input = torch.tensor(y_label)
        generator.eval()
        features = generator(y_input)['output']

        # Apply t-SNE for dimensionality reduction
        features_detached = features.detach().numpy()
        tsne = TSNE(n_components=2, random_state=100)
        reduced_features = tsne.fit_transform(features_detached)

        color_mapping = sns.color_palette("viridis", n_colors=len(set(y_label)))

        # Create a scatter plot with color-coded points
        plt.figure(figsize=(10, 8))
        sns.scatterplot(x=reduced_features[:, 0], y=reduced_features[:, 1], hue=y_label, palette=color_mapping, legend=False)
        plt.title(f' Global Round {self.glob_iter}', fontsize=24)
        # plt.xlabel('t-SNE Dimension 1')
        # plt.ylabel('t-SNE Dimension 2')
        # plt.legend(loc='upper right')
        self.calculate_dist(y, y_label, reduced_features)
        plt.savefig(path, bbox_inches='tight')
        # plt.show()

    def calculate_dist(self,y, y_label, reduced_features):
        dist={}
        xx = {}
        yy = {}
        intra_classdis={}
        for i in y:
            dist[f'{i}']= 0
            xx[f'{i}'] = 0
            yy[f'{i}'] = 0

        for index, value in np.ndenumerate(y_label):
            a = reduced_features[index, 0]
            b = reduced_features[index, 1]
            xx[f'{value}'] += reduced_features[index, 0]
            yy[f'{value}'] += reduced_features[index, 1]

        for i in y:
            xx[f'{i}'] = xx[f'{i}'] / self.repeats  #中心坐标
            yy[f'{i}'] = yy[f'{i}'] / self.repeats

        for index, value in np.ndenumerate(y_label):
            dist[f'{value}'] += math.sqrt((reduced_features[index, 0] - xx[f'{value}'])**2 + (reduced_features[index, 1] - yy[f'{value}'])**2)

        all_dis = 0
        for i in y:
            intra_classdis[f'{i}'] = dist[f'{i}']/self.repeats
            all_dis += intra_classdis[f'{i}']

        print(f'round:{self.glob_iter} ,intra-class average distance:{all_dis/len(y)} ')
        coordinates_xx = np.array([xx[f'{i}'] for i in y])
        coordinates_yy = np.array([yy[f'{i}'] for i in y])
        coordinates = np.column_stack((coordinates_xx, coordinates_yy))

        distances = np.linalg.norm(coordinates[:, np.newaxis, :] - coordinates, axis=2)

        # 计算平均距离
        average_distance = np.mean(distances)

        print(f'round:{self.glob_iter} ,inter-class average distance:{average_distance} ')



