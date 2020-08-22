import argparse
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.backends.cudnn as cudnn
import os
import os.path as osp
# from tensorboardX import SummaryWriter

# from model.image_branch_after_fc_gap import ImageBranch
# from model.image_branch_before import ImageBranch as IB
from model.single_vgg_voc_split import Our_Model
# from dataset.dataset_image import prepare_for_train_image_dataloader
from dataset.dataset_pixel_split import prepare_for_train_pixel_dataloader, prepare_for_train_weak_pixel_dataloader, \
    prepare_for_val_weak_pixel_dataloader, prepare_for_val_pixel_dataloader

from util import delete_superfluous_model, get_model_path

split = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
RESTORE_FROM_WHERE = "pretrained"
EMBEDDING = "all"
lambdaa = 0.2

BATCH_SIZE = 9
NUM_WORKERS = 3
ITER_SIZE = 1
IGNORE_LABEL = 255
INPUT_SIZE = "512,512"
LEARNING_RATE = 1e-4
MOMENTUM = 0.9
NUM_CLASSES = 21
NUM_EPOCHS = 50
POWER = 0.9
RANDOM_SEED = 1234
SAVE_NUM_IMAGES = 2
SAVE_PRED_EVERY = 500
WEIGHT_DECAY = 0.0005
LOG_DIR = "./log"
weak_size = BATCH_SIZE
weak_proportion = 0.2

DATA_PATH = "data/voc2012/"
PRETRAINED_OUR_PATH = "model/segmentation/pretrained/our_qfsl_confidence"
SNAPSHOT_PATH = "model/segmentation/snapshots/vgg/lambda_split_single_1"
PATH = "output/"


DATAROOT = PATH + DATA_PATH
SNAPSHOT_DIR = PATH + SNAPSHOT_PATH + "/" + EMBEDDING
RESULT_DIR = PATH + SNAPSHOT_PATH + "/" + "result.txt"


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS,
                        help="number of workers for multithread dataloading.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--iter-size", type=int, default=ITER_SIZE,
                        help="Accumulate gradients for ITER_SIZE iterations.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of source images.")
    parser.add_argument("--is-training", action="store_true",
                        help="Whether to updates the running means and variances during the training.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--not-restore-last", action="store_true",
                        help="Whether to not restore last (FC) layers.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-epochs", type=int, default=NUM_EPOCHS,
                        help="Number of training epochs.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--random-mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--restore-from-where", type=str, default=RESTORE_FROM_WHERE,
                        help="Where restore model parameters from pretrained or saved.")
    parser.add_argument("--save-num-images", type=int, default=SAVE_NUM_IMAGES,
                        help="How many images to save.")
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--cpu", action="store_true", help="choose to use cpu device.")
    parser.add_argument("--tensorboard", action="store_true", help="choose whether to use tensorboard.")
    parser.add_argument("--log-dir", type=str, default=LOG_DIR,
                        help="Path to the directory of log.")
    parser.add_argument("--dataroot", type=str, default=DATAROOT,
                        help="Path to the file listing the data.")
    return parser.parse_args()


args = get_arguments()
device = torch.device("cuda:0" if not args.cpu else "cpu")


def lr_poly(base_lr, iter_, max_iter, power):
    return base_lr * ((1 - float(iter_) / max_iter) ** power)


def adjust_learning_rate(optimizer, i_iter, num_steps, times=1):
    lr = lr_poly(args.learning_rate, i_iter, num_steps, args.power)
    optimizer.param_groups[0]["lr"] = lr * times


def qfsl_loss(pred, mask, ignore_index=255):
    pred = torch.softmax(pred, 1)
    tmp = pred[:, 15:, :, :]
    loss = torch.sum(tmp, dim=1)
    loss = - torch.log(loss + 0.00001)
    loss = torch.mean(loss[mask != ignore_index])
    return loss


def relabel(mask):
    for i in range(5):
        mask[mask == i + 15] = i
    return mask


def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def per_class_iu(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))


def get_pseudo(pre, mask, p):
    k = torch.zeros(20).cuda()
    mc = {}
    for i in range(20):
        mc[i] = torch.zeros(0).cuda()

    for j in range(pre.shape[0]):
        lp = torch.argmax(pre[j], 0)
        lp = lp[mask[j] != 255]
        mp, _ = torch.max(pre[j], 0)
        mp = mp[mask[j] != 255]
        for i in range(20):
            tmp = mp[lp == i]
            tmp = torch.reshape(tmp, [-1])
            mc[i] = torch.cat([mc[i], tmp], -1)

    for i in range(20):
        tmp, _ = torch.sort(mc[i], descending=True)
        ind = int(p * mc[i].shape[0])
        if tmp.shape[0] == 0:
            tmp = torch.tensor([0.20])
        k[i] = -torch.log(tmp[ind])
        if i < 15:
            k[i] = -np.log(0.99999)

    k = torch.exp(-k)
    pre = pre.permute([0, 2, 3, 1])
    pre = pre / k
    pre = pre.permute([0, 3, 1, 2])
    ind = torch.argmax(pre, 1)
    value, _ = torch.max(pre, 1)
    ind[value < 1] = 255
    ind[mask == 255] = 255

    return torch.tensor(ind).long()


def main():
    """Create the model and start the training."""

    w, h = args.input_size.split(",")
    input_size = (int(w), int(h))

    cudnn.enabled = True
    best_result = {"miou": 0, "miou_t": 0, "miou_s": 0, "iter": 0, "lr": args.learning_rate}

    # Create network
    if args.restore_from_where == "pretrained":
        model = Our_Model(split)

        i_iter = 0
    else:
        restore_from = get_model_path(args.snapshot_dir)
        model_restore_from = restore_from["model"]
        i_iter = restore_from["step"]

        model = Our_Model(split)
        saved_state_dict = torch.load(model_restore_from)
        model.load_state_dict(saved_state_dict)

    cudnn.benchmark = True

    # init

    model.train()
    model.to(device)

    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)

    train_strong_loader = prepare_for_train_pixel_dataloader(dataroot=args.dataroot,
                                                             bs_train=args.batch_size,
                                                             input_size=input_size, shuffle=True, split=split)
    train_weak_loader = prepare_for_train_weak_pixel_dataloader(dataroot=args.dataroot,
                                                                bs_train=weak_size,
                                                                input_size=input_size, shuffle=True, split=split)
    test_weak_loader = prepare_for_val_weak_pixel_dataloader(dataroot=args.dataroot,
                                                             bs_val=1,
                                                             input_size=input_size, shuffle=False, split=split)
    test_loader = prepare_for_val_pixel_dataloader(dataroot=args.dataroot,bs_val=1,
                                                   input_size=input_size, shuffle=False, split=split)

    data_len = len(train_strong_loader)
    num_steps = data_len * args.num_epochs

    optimizer = optim.SGD(model.optim_parameters_1x(args),
                          lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer.zero_grad()

    optimizer_10x = optim.SGD(model.optim_parameters_10x(args),
                              lr=10 * args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer_10x.zero_grad()

    seg_loss = nn.CrossEntropyLoss(ignore_index=255)

    interp = nn.Upsample(size=(input_size[1], input_size[0]), mode="bilinear", align_corners=True)

    with open(RESULT_DIR, "a") as f:
        f.write(SNAPSHOT_PATH.split("/")[-1] + "\n")
        f.write("lambda : " + str(lambdaa) + "\n")
    for epoch in range(args.num_epochs):
        train_strong_iter = enumerate(train_strong_loader)
        train_weak_iter = enumerate(train_weak_loader)

        model.train()
        for i in range(data_len):

            loss_pixel = 0
            loss_pixel_value = 0

            optimizer.zero_grad()
            adjust_learning_rate(optimizer, i_iter, num_steps, times=1)

            optimizer_10x.zero_grad()
            adjust_learning_rate(optimizer_10x, i_iter, num_steps, times=10)

            # train strong
            try:
                _, batch = train_strong_iter.__next__()
            except StopIteration:
                train_strong_iter = enumerate(train_strong_loader)
                _, batch = train_strong_iter.__next__()

            images, masks = batch["image"], batch["label"]
            images = images.to(device)
            masks = masks.long().to(device)
            pred = model(images, "all")

            pred = interp(pred)
            loss_pixel = seg_loss(pred, masks)
            loss_qfsl = qfsl_loss(pred, masks) * (1 - weak_proportion) * lambdaa
            loss = loss_pixel + loss_qfsl

            max_ = torch.argmax(pred, 1)
            print("{} {}".format(max_[0, 200, 200].data, masks[0, 200, 200].data))

            # pred_pixel = torch.mean(torch.stack([pred_w2v, pred_ft]), 0)

            loss.backward()

            loss_pixel_value += loss.item()

            # train weak
            try:
                _, batch = train_weak_iter.__next__()
            except StopIteration:
                train_weak_iter = enumerate(train_weak_loader)
                _, batch = train_weak_iter.__next__()

            images, masks = batch["image"], batch["label"]
            images = images.to(device)
            masks = masks.long().to(device)
            pred = model(images)

            pred = interp(pred)
            loss_qfsl = qfsl_loss(pred, masks) * weak_proportion * lambdaa

            loss_qfsl.backward()

            optimizer.step()
            optimizer_10x.step()

            print("iter = {0:8d}/{1:8d},  loss_pixel = {2:.3f}".format(i_iter, num_steps, loss))

            # save model with max miou
            if i_iter % args.save_pred_every == 0 and i_iter != best_result["iter"]:
                # zsl
                hist = np.zeros((5, 5))
                model.eval()
                for index, batch in enumerate(test_weak_loader):
                    if index % 10 == 0:
                        print("\r", index, end="")

                    images, labels, size = batch["image"], batch["label"], batch["size"]
                    w, h = list(map(int, size[0].split(",")))
                    interp_val = nn.Upsample(size=(h, w), mode="bilinear", align_corners=True)

                    images = images.to(device)
                    labels = relabel(labels).numpy()
                    # labels = labels.numpy()
                    pred = model(images, "weak")
                    pred = interp_val(pred)

                    pred = pred[0].permute(1, 2, 0)
                    pred = torch.max(pred, 2)[1].byte()
                    pred_cpu = pred.data.cpu().numpy()
                    hist += fast_hist(labels.flatten(), pred_cpu.flatten(), 5)

                mIoUs = per_class_iu(hist)
                print(mIoUs)
                mIoU = round(np.nanmean(mIoUs) * 100, 2)
                print(mIoU)

                # gzsl
                hist_g = np.zeros((20, 20))
                for index, batch in enumerate(test_loader):
                    if index % 10 == 0:
                        print("\r", index, end="")

                    images, labels, size = batch["image"], batch["label"], batch["size"]
                    w, h = list(map(int, size[0].split(",")))
                    interp_val = nn.Upsample(size=(h, w), mode="bilinear", align_corners=True)

                    images = images.to(device)
                    labels = labels.numpy()
                    pred = model(images,"all")
                    pred = interp_val(pred)

                    pred = pred[0].permute(1, 2, 0)
                    pred = torch.max(pred, 2)[1].byte()
                    pred_cpu = pred.data.cpu().numpy()
                    hist_g += fast_hist(labels.flatten(), pred_cpu.flatten(), 20)

                mIoUs_g = per_class_iu(hist_g)
                print(mIoUs_g)
                mIoU_t = round(np.nanmean(mIoUs_g[15:]) * 100, 2)
                mIoU_s = round(np.nanmean(mIoUs_g[:15]) * 100, 2)
                print(mIoU_s)
                print(mIoU_t)

                if mIoU_t > best_result["miou_t"]:
                    print("taking snapshot ...")
                    torch.save(model.state_dict(), osp.join(args.snapshot_dir, str(i_iter) + "_model.pth"))
                    delete_superfluous_model(args.snapshot_dir, 1)

                    best_result = {"miou_s": mIoU_s, "miou_t": mIoU_t, "miou": mIoU, "iter": i_iter}
                with open(RESULT_DIR, "a") as f:
                    f.write("i_iter:{:d}\tmiou:{:0.5f}\tmiou_s:{:0.5f}\tmiou_t:{:0.5f}\tbest_result:{}\n".format(
                        i_iter, mIoU, mIoU_s, mIoU_t, best_result))

            i_iter += 1


if __name__ == "__main__":
    main()
