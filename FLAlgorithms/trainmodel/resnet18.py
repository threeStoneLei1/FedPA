import torch.nn.functional as F
import torchvision.models as models
from torch import nn
from utils.model_config import CONFIGS_
class ResNet18(nn.Module):
    def __init__(self, dataset='cifar10'):
        super(ResNet18, self).__init__()
        print("Creating model for {}".format(dataset))
        self.dataset = dataset
        configs, input_channel, self.output_dim, self.hidden_dim, self.latent_dim = CONFIGS_[dataset]
        print('Network configs:', configs)
        self.feature_extractor = models.resnet18(num_classes=self.latent_dim)
        self.classifier =  nn.Sequential(
                                nn.Linear(self.latent_dim, self.output_dim))

    def forward(self, x, start_layer_idx=0, logit=True):
        if start_layer_idx < 0:
            z = self.classifier(x)
            out = F.log_softmax(z, dim=1)
            result = {'output': out}
            if logit:
                result['logit'] = z
            return result


        results = {}

        f = self.feature_extractor(x)
        z = self.classifier(f)

        if self.output_dim > 1:
            results['output'] = F.log_softmax(z, dim=1)
        else:
            results['output'] = z
        if logit:
            results['logit'] = z
        results['samples_prototype'] = f
        return results