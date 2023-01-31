import torch


def l2_norm(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)

    normp = torch.sum(buffer, 1).add_(1e-12)
    norm = torch.sqrt(normp)

    _output = torch.div(input, norm.view(-1, 1).expand_as(input))
    output = _output.view(input_size)
    return output

class KNearestNeigbors(torch.nn.Module):
    def __init__(
            self,
            number_of_neighbours,
            embedding_collection,
            labels_int
    ):
        """
        # Google + Stackoverflow + ChatGPT

        Purpose of KNN is to classify data points
        by clustering the inference data point (X) by its neigbors (*),
        where a neigbor means a single data point where we know the label.

            (** ...*)
            (** X ...*) X belongs to this cluster

                    (** ...*)
                    (** ...*)

        Create a list of embedding vectors from training set images -> raw collection: [1000, 256]
        Check the similarity between X (embedding vector of the inference image)
        and raw collection [1, 256]
        Look at the similarities and find the most similar K neighbors to X.
        Look at the labels of the most similar K neigbors.
        Determine X as the label of most similar K neighbors.
        """
        super().__init__()

        self.raw_collection = embedding_collection  # training dataset vectors
        self.labels_int = labels_int # training dataset labels of each vector as integer
        self.K = number_of_neighbours  # number of neigbbors that will be used for KNN.

    def forward(self, embedding):
        # calculate cosine similarities
        cos_sim = torch.nn.functional.linear(l2_norm(self.raw_collection),
                                             l2_norm(embedding)[0])

        # find the most similar (top) K vectors.
        cos_sim_topK = cos_sim.topk(1 + self.K)

        # first index of cos_sim_topK is indexes of these most similar neighbors.
        indexes = cos_sim_topK[1][1:self.K]

        # second index of cos_sim_topK is the similarity scores of these most similar neighbors.
        probs = cos_sim_topK[0][1:self.K]

        preds_int = self.labels_int[indexes]

        unqs, counts = torch.unique(preds_int, return_counts=True)
        pred_single_int = unqs[counts.argmax()]

        neighbour_confidence = torch.max(counts) / torch.sum(counts)
        index_cond = preds_int == pred_single_int
        confidence = probs[index_cond.nonzero()][0]

        return pred_single_int, confidence, neighbour_confidence