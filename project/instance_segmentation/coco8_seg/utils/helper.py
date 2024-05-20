import os
import json
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms.v2 import functional as F
from typing import List, Dict

def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]

    fig, axs = plt.subplots(ncols = len(imgs), squeeze = False)
    
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels = [], yticklabels = [], xticks = [], yticks = [])

def save_loss_curve(result: Dict[str, List[float]], dir_name: str):
    epochs = range(len(result['total_train_loss']))
    plt.figure(figsize = (15, 10))

    loss_types = ['total_train_loss', 'total_train_loss_classifier', 'total_train_loss_box_reg', 'total_train_loss_mask', 'total_train_loss_objectness', 'total_train_loss_rpn_box_reg']
    loss_titles = ['Loss', 'Loss Classifier', 'Loss Box Reg', 'Loss Mask', 'Loss Objectness', 'Loss RPN Box Reg']

    for i, loss_type in enumerate(loss_types):
        plt.subplot(2, 3, i + 1)
        plt.plot(epochs, result[loss_type], label = 'train')
        plt.plot(epochs, result['total_val_' + loss_type.split('_train_')[1]], label = 'val')
        plt.title(loss_titles[i])
        plt.xlabel('Epochs')
        plt.legend(loc = 'upper right')

    os.makedirs('result/' + dir_name, exist_ok = True)

    plt.savefig(f'result/{dir_name}/loss_plot.png')

    with open(f'result/{dir_name}/data.json', 'w') as f:
        json.dump(result, f)

    plt.close()