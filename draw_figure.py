import os
import glob
import matplotlib.pyplot as plt
import cv2
import numpy as np
# import matplotlib.pyplot as plt2


log_folder = "/media/data/vuba/tsn/mmaction/work_dirs/3_views_split1/"
save_img_name = log_folder.split('/')[-2] + ".jpg"
print(save_img_name)
save = log_folder + "figures/"
if not os.path.exists(save):
    os.makedirs(save)
log_file = glob.glob(log_folder + "*.log")

epoch_nums = []
acc_top1s = []

epoch_trains = []
loss_change = []
learning_rates = []
if len(log_file) > 1:
    print("More than one log file")
    exit(1)
else:
    log = log_file[0]
    f = open(log, "r")
    lines = f.readlines()
    for line in lines:
        if 'Epoch' in line and '(val)' in line:
            segs = line.split(' ')
            epoch_num = int(segs[6].split('[')[1].split(']')[0])
            epoch_nums.append(epoch_num)
            acc_top1 = float(segs[-4][0:-1])
            acc_top1s.append(acc_top1)
            # print(epoch_num)
            # print(acc_top1)
            # print(segs[6] + "   " + segs[-4])
        elif 'Epoch' in line and '(val)' not in line and 'loss_cls' in line:
            segs = line.split(' ')
            learning_rates.append(float(segs[7][0:-1]))
            epoch_num = int(segs[6].split('[')[1].split(']')[0])
            loss = float(segs[-1].strip())
            epoch_trains.append(epoch_num)
            loss_change.append(loss)
            # print(loss)
    max_acc = max(acc_top1s)
    best_checkpoint = acc_top1s.index(max_acc)

    plt.plot(epoch_nums, acc_top1s, 'r-')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy over Epochs ( MAX = ' + str(max_acc) +", at epoch: " + str(best_checkpoint) + " )" )
    plt.savefig(save + 'acc_vs_epochs.jpg')
    plt.close()

    plt.plot(epoch_trains, loss_change, 'b-')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss change')
    plt.savefig(save + 'loss_vs_epochs.jpg')
    plt.close()

    plt.plot(epoch_trains, learning_rates, 'g-')
    plt.xlabel('Epochs')
    plt.ylabel('Learning rates')
    plt.title('Learning rate change')
    plt.savefig(save + 'lr_vs_epochs.jpg')
    plt.close()

img1 = cv2.imread(save + 'acc_vs_epochs.jpg')
print("img1 shape ", img1.shape)
img2 = cv2.imread(save + 'loss_vs_epochs.jpg')
img3 = cv2.imread(save + 'lr_vs_epochs.jpg')
vis = np.concatenate((img1, img2), axis=1)
vis = np.concatenate((vis, img3), axis=1)
cv2.imwrite(save + save_img_name, vis)
