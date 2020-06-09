from argparse import ArgumentParser
import os
import tensorflow as tf
import random
from matplotlib import pyplot as plt
import numpy as np
import math
import os
from pathlib import Path

import torch
from torch import optim
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms
import models
from loader import get_loader, get_data_path
from models import get_model
from augmentations import *
from models.vaeIdsiaStnTF import create_encoder_model, CVAE, train_step, get_optimizer, compute_loss_tf

USE_CUDA = True
try:
    torch.cuda.current_device()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
except:
    USE_CUDA = False

# Setup
parser = ArgumentParser(description='Variational Prototyping Encoder (VPE)')
parser.add_argument('--seed',       type=int,   default=42,             help='Random seed')
parser.add_argument('--arch',       type=str,   default='vaeIdsia',  help='network type: vaeIdsia, vaeIdsiaStn')
parser.add_argument('--dataset',    type=str,   default='gtsrb2TT100K', help='dataset to use [gtsrb, gtsrb2TT100K, belga2flickr, belga2toplogo]')
parser.add_argument('--exp',        type=str,   default='exp_list',     help='training scenario')
parser.add_argument('--resume',     type=str,   default=None,           help='Resume training from previously saved model')

parser.add_argument('--epochs',     type=int,   default=2000,           help='Training epochs')
parser.add_argument('--lr',         type=float, default=1e-4,           help='Learning rate')
parser.add_argument('--batch_size', type=int,   default=64,            help='Batch size')

parser.add_argument('--img_cols',   type=int,   default=64,             help='resized image width')
parser.add_argument('--img_rows',   type=int,   default=64,             help='resized image height')
parser.add_argument('--workers',    type=int,   default=4,              help='Data loader workers')

args = parser.parse_args()

random.seed(args.seed)
torch.manual_seed(args.seed)
plt.switch_backend('agg')  # Allow plotting when running remotely

save_epoch = 100 # save log images per save_epoch

# 02 rotation + flip augmentation option
# Setup Augmentations
data_aug_tr= Compose([Scale(args.img_cols), # resize longer side of an image to the defined size
                      CenterPadding([args.img_rows, args.img_cols]), # zero pad remaining regions
                      RandomHorizontallyFlip(), # random horizontal flip
                      RandomRotate(180)])  # ramdom rotation

data_aug_te= Compose([Scale(args.img_cols), 
                     CenterPadding([args.img_rows, args.img_cols])])

result_path = 'results_' + args.dataset
if not os.path.exists(result_path):
  os.makedirs(result_path)
outimg_path =  "./img_log_" + args.dataset
if not os.path.exists(outimg_path):
  os.makedirs(outimg_path)

f_loss = open(os.path.join(result_path, "log_loss.txt"),'w')
f_loss.write('Network type: %s\n'%args.arch)
f_loss.write('Learning rate: %05f\n'%args.lr)
f_loss.write('batch-size: %s\n'%args.batch_size)
f_loss.write('img_cols: %s\n'%args.img_cols)
f_loss.write('Augmentation type: flip, centercrop\n\n')
f_loss.close()

f_iou = open(os.path.join(result_path, "log_acc.txt"),'w')
f_iou.close()

f_iou = open(os.path.join(result_path, "log_val_acc.txt"),'w')
f_iou.close()

# set up GPU
# we could do os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Data
data_loader = get_loader(args.dataset)
data_path = get_data_path(args.dataset)

tr_loader = data_loader(data_path, args.exp, is_transform=True, split='train', img_size=(args.img_rows, args.img_cols), augmentations=data_aug_tr)
te_loader = data_loader(data_path, args.exp, is_transform=True, split='test', img_size=(args.img_rows, args.img_cols), augmentations=data_aug_te)

trainloader = DataLoader(tr_loader, batch_size=args.batch_size, num_workers=args.workers, shuffle=True, pin_memory=True)
testloader = DataLoader(te_loader, batch_size=args.batch_size, num_workers=args.workers, shuffle=True, pin_memory=True)

# define model or load model
net = get_model(args.arch, n_classes=None)
if USE_CUDA:
  net.cuda()

if args.resume is not None:
  pre_params = torch.load(args.resume)
  net.init_params(pre_params)

reconstruction_function = nn.BCELoss()
reconstruction_function.reduction = 'sum'
def loss_function(recon_x, x, mu, logvar):
    BCE = reconstruction_function(recon_x, x)
    # x_for_tf = x.data.numpy().transpose((0,2,3,1))
    # recon_x_for_tf = recon_x.data.numpy().transpose((0,2,3,1))
    # mu_for_tf = mu.data.numpy()
    # logvar_for_tf = logvar.data.numpy()
    #
    #
    # BCE_tf = tf.keras.losses.BinaryCrossentropy()(x_for_tf, recon_x_for_tf)
    # KLD_tf = -0.5*tf.reduce_sum(1+ logvar_for_tf - (tf.pow(mu_for_tf,2) + tf.exp(logvar_for_tf)))

    # tf.nn.sigmoid_cross_entropy_with_logits(recon_x_for_tf,x_for_tf)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)
    return BCE + KLD

# Construct optimiser
optimizer = optim.Adam(net.parameters(), lr=args.lr) # 1e-4

num_train = len(tr_loader.targets)
num_test = len(te_loader.targets)
batch_iter = math.ceil(num_train/args.batch_size)
batch_iter_test = math.ceil(num_test/args.batch_size)


tf_model = CVAE(latent_dim=300)
tf_optimizer = get_optimizer()

def train(e):
  n_classes = tr_loader.n_classes
  n_classes_te = te_loader.n_classes

  print('start train epoch: %d'%e)
  net.train()
  # tf_encoder_model = models.vaeIdsiaStnTF.create_encoder_model()
  for i, (input, target, template) in enumerate(trainloader):

    optimizer.zero_grad()
    target = torch.squeeze(target)

    if USE_CUDA:
      input, template = input.cuda(non_blocking=True), template.cuda(non_blocking=True)



    recon, mu, logvar, input_stn = net(input)


    # convert to tensorflow ordering:
    if USE_CUDA:
      data_for_tf = input.cpu().data.numpy().transpose((0, 3, 2, 1))
      target_for_tf = template.cpu().data.numpy().transpose((0, 3, 2, 1))
    else:
      data_for_tf = input.data.numpy().transpose((0, 3, 2, 1))
      target_for_tf =template.data.numpy().transpose((0, 3, 2, 1))
    # mean, logvar = tf_model.encode(data_for_tf)
    # res = tf_encoder_model(data_for_tf)

    # mean, logvar = tf_model.encode(data_for_tf)
    # z = tf_model.reparameterize(mean, logvar)
    # predictions = tf_model.sample(z)
    #
    # for i in range(100):
    #   train_step(model=tf_model,x=data_for_tf, target=target_for_tf,optimizer=tf_optimizer)
    #
    #   mean, logvar = tf_model.encode(data_for_tf)
    #   z = tf_model.reparameterize(mean, logvar)
    #   predictions = tf_model.sample(z)
    # print('yo')


    # print("res.shape", res.shape)
    loss = loss_function(recon, template, mu, logvar) # reconstruction loss

    # loss_tf = compute_loss_tf(recon_x=recon.data.numpy().transpose((0, 2, 3, 1)),
    #                           x=template.data.numpy().transpose((0, 2, 3, 1)),
    #                           mean=mu.data.numpy(),
    #                           logvar=logvar.data.numpy())
    loss_tf = train_step(model=tf_model, x=data_for_tf, target=target_for_tf, optimizer=tf_optimizer)

    # compute_loss_tf()
    print(f'Epoch:{e}  Batch:{i}/{batch_iter}  loss:{loss.data/input.numel()} tf_loss:{loss_tf}')
   
    f_loss = open(os.path.join(result_path, "log_loss.txt"),'a')
    f_loss.write('Epoch:%d  Batch:%d/%d  loss:%08f\n'%(e, i, batch_iter, loss.data/input.numel()))
    f_loss.close()



    tf_model.save_weights(f'/Users/wrytl/save_model.h5')
    loss.backward()
    optimizer.step()

    if i < 1 and (e%save_epoch == 0):
      out_folder =  "%s/Epoch_%d_train"%(outimg_path, e)
      out_root = Path(out_folder)
      if not out_root.is_dir():
        os.mkdir(out_root)

      torchvision.utils.save_image(input.data, '{}/batch_{}_data.jpg'.format(out_folder,i), nrow=8, padding=2)
      torchvision.utils.save_image(input_stn.data, '{}/batch_{}_data_stn.jpg'.format(out_folder, i), nrow=8, padding=2) 
      torchvision.utils.save_image(recon.data, '{}/batch_{}_recon.jpg'.format(out_folder,i), nrow=8, padding=2)
      torchvision.utils.save_image(template.data, '{}/batch_{}_target.jpg'.format(out_folder,i), nrow=8, padding=2)

  if e%save_epoch == 0:
    class_target = torch.LongTensor(list(range(n_classes)))
    class_template = tr_loader.load_template(class_target)
    if USE_CUDA:
      class_template = class_template.cuda(non_blocking=True)
    with torch.no_grad():
      class_recon, class_mu, class_logvar, _ = net(class_template)
    tf_model.save_weights(f'/Users/wrytl/save_model.h5')
    torchvision.utils.save_image(class_template.data, '{}/templates.jpg'.format(out_folder), nrow=8, padding=2)  
    torchvision.utils.save_image(class_recon.data, '{}/templates_recon.jpg'.format(out_folder), nrow=8, padding=2) 
  
def score_NN(pred, class_feature, label, n_classes):

  sample_correct = torch.zeros(n_classes)
  sample_all = torch.zeros(n_classes)
  sample_rank = torch.zeros(n_classes, n_classes) # rank per class
  sample_distance = torch.ones(pred.shape[0], n_classes)*math.inf

  pred = pred.data.cpu() # batch x latent size
  class_feature = class_feature.data.cpu() # n_classes x latent size
  label = label.numpy()
  for i in range(n_classes):
    cls_feat = class_feature[i,:]
    cls_mat = cls_feat.repeat(pred.shape[0],1)
    # euclidean distance
    sample_distance[:,i] = torch.norm(pred - cls_mat,p=2, dim=1)
  
  sample_distance = sample_distance.cpu().numpy()
  indices = np.argsort(sample_distance, axis=1) # sort ascending order

  for i in range(indices.shape[0]):
    rank = np.where(indices[i,:] == label[i])[0][0] # find rank
    sample_rank[label[i]][rank:] += 1 # update rank 
    sample_all[label[i]] += 1 # count samples per class
    if rank == 0:
      sample_correct[label[i]] += 1 # count rank 1 (correct classification)

  return sample_correct, sample_all, sample_rank


mean_scores = []
mean_rank = []
def test(e, best_acc):
  n_classes = te_loader.n_classes
  print('start test epoch: %d'%e)
  net.eval()
  accum_all = torch.zeros(n_classes)
  rank_all = torch.zeros(n_classes, n_classes) # rank per class
  accum_class = torch.zeros(n_classes)

  # get template latent z
  class_target = torch.LongTensor(list(range(n_classes)))
  class_template = te_loader.load_template(class_target)
  if USE_CUDA:
    class_template = class_template.cuda(non_blocking=True)
  with torch.no_grad():
    class_recon, class_mu, class_logvar, _ = net(class_template)
  
  for i, (input, target, template) in enumerate(testloader):

    target = torch.squeeze(target)
    if USE_CUDA:
      input, template = input.cuda(non_blocking=True), template.cuda(non_blocking=True)
    with torch.no_grad():
      recon, mu, logvar, input_stn  = net(input)
    
    sample_correct, sample_all, sample_rank = score_NN(mu, class_mu, target, n_classes)
    accum_class += sample_correct
    accum_all += sample_all
    rank_all = rank_all + sample_rank # [class_id, topN]
    
    print('Epoch:%d  Batch:%d/%d  processing...'%(e, i, batch_iter_test))

    if i < 1 and (e%save_epoch == 0):
      out_folder =  "%s/Epoch_%d_test"%(outimg_path, e)
      out_root = Path(out_folder)
      if not out_root.is_dir():
        os.mkdir(out_root)

      torchvision.utils.save_image(input.data, '{}/batch_{}_data.jpg'.format(out_folder,i), nrow=8, padding=2)
      torchvision.utils.save_image(input_stn.data, '{}/batch_{}_data_stn.jpg'.format(out_folder, i), nrow=8, padding=2) 
      torchvision.utils.save_image(recon.data, '{}/batch_{}_recon.jpg'.format(out_folder,i), nrow=8, padding=2)
      torchvision.utils.save_image(template.data, '{}/batch_{}_target.jpg'.format(out_folder,i), nrow=8, padding=2)

  if e%save_epoch == 0:
    torchvision.utils.save_image(class_template.data, '{}/templates.jpg'.format(out_folder), nrow=8, padding=2)  
    torchvision.utils.save_image(class_recon.data, '{}/templates_recon.jpg'.format(out_folder), nrow=8, padding=2)  

  acc_all = accum_class.sum() / accum_all.sum() 
  acc_cls = torch.div(accum_class, accum_all)
  
  rank_sample_avg = rank_all.sum(0) / accum_all.sum() # [class_id, topN]
  rank_cls = torch.div(rank_all, torch.transpose(accum_all.expand_as(rank_all),0,1))
  rank_cls_avg = torch.mean(rank_cls,dim=0)



  # write result part
  acc_trcls = torch.gather(acc_cls, 0, te_loader.tr_class)
  acc_tecls =torch.gather(acc_cls, 0, te_loader.te_class)

  print('========epoch(%d)========='%e)
  print('Seen Classes')
  for i, class_acc in enumerate(acc_trcls):
    print('cls:%d  acc:%02f'%(te_loader.tr_class[i], class_acc))
  print('Unseen Classes')
  for i, class_acc in enumerate(acc_tecls):
    print('cls:%d  acc:%02f'%(te_loader.te_class[i], class_acc))
  print('====================================')
  print('acc_avg:%02f'%acc_all)
  print('acc_cls:%02f'%acc_cls.mean())
  print('acc_trcls:%02f'%acc_trcls.mean())
  print('acc_tecls:%02f'%acc_tecls.mean())
  print('rank sample avg: %02f'%rank_sample_avg.mean())
  print('rank cls avg: %02f'%rank_cls_avg.mean())
  print('====================================')

  f_iou = open(os.path.join(result_path, "log_acc.txt"),'a')
  f_iou.write('epoch(%d), acc_cls: %04f  acc_trcls: %04f  acc_tecls: %04f  acc_all: %04f  top3: %04f  top5: %04f\n'%(e, acc_cls.mean(), acc_trcls.mean(), acc_tecls.mean(), acc_all, rank_sample_avg[2], rank_sample_avg[4]))
  f_iou.close()


  # if best_acc < acc_all: # update best score
  #   best_acc = acc_all
  if best_acc < acc_tecls.mean(): # update best score

    f_iou_class = open(os.path.join(result_path, "best_iou.txt"),'w')
    f_rank = open(os.path.join(result_path, "best_rank.txt"),'w')
    torch.save(net.state_dict(), os.path.join('%s_testBest_net.pth'%args.dataset))

    best_acc = acc_tecls.mean()
    f_iou_class.write('Best score epoch:  %d\n'%e)
    f_iou_class.write('acc cls: %.4f  acc all: %.4f  rank mean: %.4f \n'%(acc_cls.mean(), acc_all, rank_all.mean()))
    f_iou_class.write('acc tr cls: %.4f  acc te cls: %.4f\n'%(acc_trcls.mean(), acc_tecls.mean()))
    f_iou_class.write('top3: %.4f  top5: %.4f\n'%(rank_sample_avg[2], rank_sample_avg[4]))

    f_iou_class.write('\nSeen classes\n')
    for i, class_acc in enumerate(acc_trcls):
      f_iou_class.write('cls:%d  acc:%02f\n'%(te_loader.tr_class[i], class_acc))
    f_iou_class.write('\nUnseen classes\n')
    for i, class_acc in enumerate(acc_tecls):
      f_iou_class.write('cls:%d  acc:%02f\n'%(te_loader.te_class[i], class_acc))
    f_iou_class.close()
    
    for i, rank_acc in enumerate(rank_sample_avg):
      f_rank.write('rank sample %d: %.4f\n'%(i+1, rank_acc))
    f_rank.write('\n')
    for i, rank_acc in enumerate(rank_cls_avg):
      f_rank.write('rank cls %d: %.4f\n'%(i+1, rank_acc))
    f_rank.close()
    
  # Save weights and scores
  # if e % 100 == 0:
  #   pass
  #   torch.save(net.state_dict(), os.path.join('flickr2belga_latest_net.pth'))

  ############# Plot scores
  mean_scores.append(acc_tecls.mean())
  es = list(range(len(mean_scores)))
  plt.plot(es, mean_scores, 'b-')
  plt.xlabel('Epoch')
  plt.ylabel('Unseen mean IoU')
  plt.savefig(os.path.join(result_path, 'unseen_ious.png'))
  plt.close()

  ############# plot rank
  # mean_rank.append(rank_all.mean())
  # rank_es = list(range(len(mean_rank)))
  # plt.plot(rank_es, mean_rank, 'b-')
  # plt.xlabel('Epoch')
  # plt.ylabel('Mean rank')
  # plt.savefig(os.path.join(result_path, 'rank.png'))
  # plt.close()

  return best_acc

if __name__ == "__main__":

  out_root = Path(outimg_path)
  if not out_root.is_dir():
    os.mkdir(out_root)
  best_acc = 0
  for e in range(1, args.epochs + 1):
    train(e)
    best_acc = test(e, best_acc)
