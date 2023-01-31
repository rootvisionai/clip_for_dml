import torch
import sklearn.preprocessing


def binarize(T, nb_classes):
    T = T.cpu().numpy()
    T = sklearn.preprocessing.label_binarize(
        T, classes=range(0, nb_classes)
    )
    T = torch.FloatTensor(T)  # .cuda()
    return T

def l2_norm(x):
    input_size = x.size()
    buffer = torch.pow(x, 2)
    normp = torch.sum(buffer, 1).add_(1e-12)
    norm = torch.sqrt(normp)
    _output = torch.div(x, norm.view(-1, 1).expand_as(x))
    output = _output.view(input_size)
    return output

class LinearProjection(torch.nn.Module):
    def __init__(
            self,
            normalize=True,
            nb_classes=10,
            noise="rand",  # rand, dropout
            k=0.1,
            device="cuda"
    ):

        super().__init__()

        self.normalize = normalize
        self.nb_classes = nb_classes
        self.noise = noise
        self.k = k
        self.device = device


    def forward (
            self,
            z1: torch.Tensor,
            labels: torch.Tensor,
    ) -> torch.Tensor:

        linear = torch.nn.Linear(in_features=z1.shape[-1],
                                 out_features=z1.shape[0], bias=False)

        labels_bin = binarize(labels, self.nb_classes).to(self.device)

        # solve with exact solution and append new weights
        z1 = l2_norm(z1) if self.normalize else z1
        with torch.no_grad():
            linear.weight = torch.nn.Parameter(torch.matmul(torch.pinverse(z1), labels_bin).T.detach())

        # add augmentation to the vector
        if self.noise=="rand":
            z2 = torch.rand_like(z1)
            z2 = l2_norm(z2) if self.normalize else z2
            z2 = z1 + (z2 * self.k).detach()
            z2 = l2_norm(z2) if self.normalize else z2
        elif self.noise=="dropout":
            z2 = torch.nn.functional.dropout1d(z1, p=self.k, training=True)
            z2 = l2_norm(z2) if self.normalize else z2

        # run inference and calculate cross-entropy loss
        loss = torch.nn.functional.cross_entropy(input=linear(z2), target=labels)

        return loss