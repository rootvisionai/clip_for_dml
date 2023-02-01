from models import hugface, knn
from losses import LinearProjection
from commons import utils
from datasets import Data, get_transform


import torch
from collections import deque
import tqdm
import os


def pretrain(cfg):

    resume = f"{cfg.run_id}_{cfg.preprocessing.image_size}_{cfg.run_id}_{cfg.lp.type}_{cfg.lp.k}_{cfg.embedding_size}"
    resume = os.path.join("checkpoints", resume)

    ds_tr = Data(root=cfg.training_path, transform=get_transform(cfg, train=True))
    dl_tr = torch.utils.data.DataLoader(
        dataset=ds_tr,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=cfg.num_workers
    )

    ds_ev = Data(root=cfg.eval_path, transform=get_transform(cfg, train=False))
    dl_ev = torch.utils.data.DataLoader(
        dataset=ds_ev,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=cfg.num_workers
    )

    model = hugface.CLIP(
                loss_function=LinearProjection(
                    normalize=True,
                    nb_classes=ds_tr.nb_classes(),
                    noise=cfg.lp.type,
                    k=cfg.lp.k,
                    device=cfg.device
                ),
                embedding_size=cfg.embedding_size,
                optimizer=cfg.optimizer,
                lr=cfg.lr,
                device=cfg.device)
    last_epoch = 0

    if not os.path.isdir(resume):
        os.makedirs(resume)

    if os.path.isfile(os.path.join(resume, "ckpt.pth")):
        ckpt = torch.load(os.path.join(resume, "ckpt.pth"))
        model.load_state_dict(ckpt["state_dict"])
        last_epoch = ckpt["last_epoch"] if "last_epoch" in ckpt else 0

    model.to(cfg.device)
    model.eval()
    model.embedding.train()

    loss_hist = deque(maxlen=20)
    precisions = [0]
    for epoch in range(last_epoch, cfg.epochs):
        pbar = tqdm.tqdm(enumerate(dl_tr))
        for cnt, (image_batch, label_batch) in pbar:
            loss = model.training_step(image_batch, label_batch)
            loss_hist.append(loss)
            pbar.set_description(f"EPOCH: {epoch} | ITER: {cnt}/{len(dl_tr)} | LOSS: {torch.mean(torch.tensor(loss_hist))}")

        torch.save({
            "state_dict": model.state_dict(),
            "opt_state_dict": model.opt.state_dict(),
            "last_epoch": epoch+1
        }, os.path.join(resume, "ckpt_latest.pth"))

        if (epoch + 1) % cfg.lp.interval == 0:
            model.criterion.k = cfg.lp.k*cfg.lp.coeff
            if model.criterion.k>cfg.lp.max:
                model.criterion.k = cfg.lp.max

        if (epoch+1)%cfg.eval_interval==0:
            print("EVALUATING...")
            precision = utils.evaluate_cos(model, dl_ev)[0]
            if precision>max(precisions):
                torch.save({
                    "state_dict": model.state_dict(),
                    "opt_state_dict": model.opt.state_dict(),
                    "last_epoch": epoch + 1
                }, os.path.join(resume, "ckpt.pth"))
            print(f"PRECISION: {precision}")


def train_knn(cfg):

    resume = f"{cfg.run_id}_{cfg.preprocessing.image_size}_{cfg.run_id}_{cfg.lp.type}_{cfg.lp.k}_{cfg.embedding_size}"
    resume = os.path.join("checkpoints", resume)

    if os.path.isfile(os.path.join(resume, "ckpt_ready.pth")):
        pass
    else:
        ds_tr = Data(root=cfg.training_path, transform=get_transform(cfg, train=False))
        dl_tr = torch.utils.data.DataLoader(
            dataset=ds_tr,
            batch_size=cfg.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=cfg.num_workers
        )

        model = hugface.CLIP(
                    loss_function=LinearProjection(
                        normalize=True,
                        nb_classes=ds_tr.nb_classes(),
                        noise="rand",  # rand, dropout
                        k=cfg.lp_k,
                        device=cfg.device
                    ),
                    embedding_size=cfg.embedding_size,
                    optimizer=cfg.optimizer,
                    lr=cfg.lr,
                    device=cfg.device)

        if not os.path.isdir(resume):
            os.makedirs(resume)

        if os.path.isfile(os.path.join(resume, "ckpt.pth")):
            ckpt = torch.load(os.path.join(resume, "ckpt.pth"))
            model.load_state_dict(ckpt["state_dict"])

        model.to(cfg.device)
        model.eval()
        model.embedding.train()

        model.criterion.k = 0.3

        pbar = tqdm.tqdm(enumerate(dl_tr))
        for cnt, (image_batch, label_batch) in pbar:
            embs = model.inference(image_batch)
            if cnt==0:
                embeddings = embs
                labels = label_batch
            else:
                embeddings = torch.cat([embeddings, embs])
                labels = torch.cat([labels, label_batch])

            pbar.set_description(f"[KNN] ITER: {cnt}/{len(dl_tr)}")

        ckpt["embeddings"] = embeddings
        ckpt["labels"] = labels
        torch.save(ckpt, os.path.join(resume, "ckpt_ready.pth"))

def test_package(cfg):
    resume = f"{cfg.run_id}_{cfg.preprocessing.image_size}_{cfg.run_id}_{cfg.lp.type}_{cfg.lp.k}_{cfg.embedding_size}"
    resume = os.path.join("checkpoints", resume)

    # import dataset
    ds_ev = Data(root=cfg.eval_path, transform=get_transform(cfg, train=False))
    dl_ev = torch.utils.data.DataLoader(
        dataset=ds_ev,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=cfg.num_workers
    )

    model = hugface.CLIP(
                loss_function=LinearProjection(
                    normalize=True,
                    nb_classes=ds_ev.nb_classes(),
                    noise="rand",  # rand, dropout
                    k=cfg.lp_k,
                    device=cfg.device
                ),
                embedding_size=cfg.embedding_size,
                optimizer=cfg.optimizer,
                lr=cfg.lr,
                device=cfg.device)

    checkpoint = torch.load(os.path.join(resume, "ckpt_ready.pth"))
    model.load_state_dict(checkpoint["state_dict"])

    model.classifier = knn.KNearestNeigbors(
            number_of_neighbours=cfg.number_of_neighbours,
            embedding_collection=checkpoint["embeddings"].to(cfg.device),
            labels_int=checkpoint["labels"].to(cfg.device)
        )

    model.to(cfg.device)
    model.eval()

    with torch.no_grad():
        print("**Evaluating for sanity check ...**")
        precision = utils.evaluate_knn_model(model, dl_ev, cfg.device)
        print(F"PRECISION: {precision}")

if __name__=="__main__":
    cfg = utils.load_config("./config.yml")
    pretrain(cfg)
    train_knn(cfg)
    test_package(cfg)
