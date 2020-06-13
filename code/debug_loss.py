import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


import numpy as np
import tensorflow as tf
import torch
from torch import nn


def main():
    debug_data = np.load('debug_data.npy', allow_pickle=True).item()
    template_torch = debug_data['template_torch'].transpose((0,2,3,1))
    torch_recon = debug_data['torch_recon'].transpose((0, 2, 3, 1))
    target_for_tf = debug_data['target_for_tf']
    tf_reco = debug_data['tf_reco']
    print(tf.keras.losses.BinaryCrossentropy()(template_torch, torch_recon).numpy())
    print(tf.keras.losses.BinaryCrossentropy()(target_for_tf, tf_reco).numpy())

    reconstruction_function = nn.BCELoss()
    reconstruction_function.reduction = 'sum'
    # print(reconstruction_function(torch.from_numpy(debug_data['template_torch']).float(),
    #                               torch.from_numpy(debug_data['torch_recon']).float()))

    print(nn.BCELoss()(torch.from_numpy(debug_data['tf_reco'].transpose((0, 3, 1, 2))),
                                  torch.from_numpy(debug_data['target_for_tf'].transpose((0, 3, 1, 2)))))

    print(nn.BCELoss()(torch.from_numpy(debug_data['torch_recon']).float(),
                 torch.from_numpy(debug_data['template_torch']).float()))
    print('asasa')


if __name__ == '__main__':
    main()
