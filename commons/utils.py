import yaml, json
import types
import torch
import torch.nn.functional as F
import shutil
import tqdm
import os


def load_config(path_to_config_yaml: object = "./config.yaml") -> object:

    with open(path_to_config_yaml) as f:
        dct = yaml.safe_load(f)

    def load_object(dct):
        return types.SimpleNamespace(**dct)

    cfg = json.loads(json.dumps(dct), object_hook=load_object)

    return cfg


def l2_norm(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)

    normp = torch.sum(buffer, 1).add_(1e-12)
    norm = torch.sqrt(normp)

    _output = torch.div(input, norm.view(-1, 1).expand_as(input))
    output = _output.view(input_size)
    return output

def precision_at_k(T, Y, k):
    """
    T : [nb_samples] (target labels)
    Y : [nb_samples x k] (k predicted labels/neighbours)
    """

    true = 0
    false = 0
    for t, y in zip(T, Y):
        unqs, cnts = torch.unique(torch.Tensor(y).long()[:k], return_counts=True)
        p = unqs[cnts.argmax()]
        if t == p:
            true += 1
        else:
            false += 1
    return true / (true + false)


def predict_batchwise(model, dataloader):
    device = "cuda"
    model.eval()

    ds = dataloader.dataset
    A = [[] for i in range(len(ds[0]))]
    with torch.no_grad():
        # extract batches (A becomes list of samples)
        for batch in dataloader:
            for i, J in enumerate(batch):
                # i = 0: sz_batch * images
                # i = 1: sz_batch * labels
                # i = 2: sz_batch * indices
                if i == 0:
                    # move images to device of model (approximate device)
                    J = model(J.cuda())

                for j in J:
                    A[i].append(j)
    model.train()
    return [torch.stack(A[i]) for i in range(len(A))]


def evaluate_cos(model, dataloader):
    nb_classes = dataloader.dataset.nb_classes

    # calculate embeddings with model and get targets
    X, T = predict_batchwise(model, dataloader)
    X = l2_norm(X)

    # get predictions by assigning nearest 8 neighbors with cosine
    K = 8
    Y = []
    xs = []

    cos_sim = F.linear(X, X)
    Y = T[cos_sim.topk(1 + K)[1][:, 1:]]
    Y = Y.float().cpu()

    precision = []
    for k in [1, 2, 4, 8, 16, 32]:
        p_at_k = precision_at_k(T, Y, k)
        precision.append(p_at_k)
        print("Precision @ {} neighbors : {:.3f}".format(k, 100 * p_at_k))

    return precision

def evaluate_knn_model(model, dataloader, device='cpu'):
    if os.path.exists("wrong_preds"):
        shutil.rmtree("wrong_preds")
        os.makedirs("wrong_preds")

    model.eval()
    img_paths, gts, preds, confs = [], [], [], []

    pbar = tqdm.tqdm(enumerate(dataloader))
    for cnt, (image_batch, label_batch) in pbar:
        with torch.no_grad():
            pred, conf, n_conf = model(image_batch.to(device))
            pred = pred.item()
            conf = conf.item()
            n_conf = n_conf.item()

        confs.append([conf, n_conf])
        gts.append(label_batch.item())
        preds.append(pred)

        pbar.set_description(
            'Inference: [{}/{} ({:.0f}%)]'.format(
                cnt + 1, len(dataloader),
                100. * (cnt + 1) / len(dataloader)))

    true_predictions = 0
    false_predictions = 0
    predictions = {}
    for idx, pred in enumerate(preds):
        if pred == gts[idx]:
            true_predictions += 1
        else:
            false_predictions += 1

        predictions[idx] = {"accurate": pred == gts[idx],
                            "confidence": confs[idx][0],
                            "n_confidence": confs[idx][1],
                            "pred": pred,
                            "ground_truth": gts[idx]}

        pbar.set_description(
            'Evaluate: [{}/{} ({:.0f}%)] TRUE[{}] FALSE[{}]'.format(
                idx + 1, len(dataloader),
                100. * (idx + 1) / len(dataloader),
                true_predictions,
                false_predictions
            ))

    precision = true_predictions / (true_predictions + false_predictions)
    print("true_predictions:", true_predictions)
    print("false_predictions:", false_predictions)
    print("precision: ", precision)

    with open(f"predictions.json", "w") as fp:
        json.dump(predictions, fp, indent=4)

    return precision