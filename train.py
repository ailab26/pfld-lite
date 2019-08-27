import mxnet as mx
from models.NPFLD import NPFLD
from models.CPFLD import CPFLD
from models.BASE import BASE
from models.MSBASE import MSBASE
from models.M1BASE import M1BASE
import numpy as np
from mxnet import nd
from mxnet import autograd
import os
import sys
import math
import cv2
import argparse


def preprocess(data):
    data = (data-123.0) / 58.0
    return data

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="pfld landmarks detector")
    parser.add_argument("--output_dir", type = str, default = None)
    parser.add_argument("--pretrain_param", type = str, default = None)
    parser.add_argument("--train_data_path", type = str, default = None)
    parser.add_argument("--valid_data_path", type = str, default = None)
    parser.add_argument("--learning_rate", type = float, default = 0.0001)
    parser.add_argument("--batch_size", type = int, default = 128)
    parser.add_argument("--epoches", type = int, default = 1000)
    parser.add_argument("--gpu_ids", type = str, default = "0,1")
    parser.add_argument("--image_size", type = int, default = 112)
    parser.add_argument("--num_of_pts", type = int, default = 98)
    parser.add_argument("--model_type", type = str, default = 'NPFLD')
    parser.add_argument("--logfile_name", type = str, default = 'log.txt')
    parser.add_argument("--with_angle_loss", type = str, default = 1)
    parser.add_argument("--with_category_loss", type = int, default = 0)
    parser.add_argument("--alpha", type = float, default = 1.0)
    args = parser.parse_args()


    train_data_file = args.train_data_path   
    valid_data_file = args.valid_data_path
    output_dir = args.output_dir

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    use_gpu = None
    devices = []
    if 'None' in args.gpu_ids:
        use_gpu = False
        devices.append(mx.cpu())
    else:
        use_gpu = True
        gpu_infos = args.gpu_ids.split(',')
        for gi in gpu_infos:
            devices.append(mx.gpu(int(gi)))
    
    image_size    = args.image_size
    batch_size    = args.batch_size
    epoches       = args.epoches
    base_lr       = args.learning_rate
    pts_num       = args.num_of_pts
    alpha         = args.alpha
    model_type    = args.model_type
    with_category = args.with_category_loss
    with_angle    = args.with_angle_loss
    logF_name     = os.path.join(output_dir, args.logfile_name)

    logFile    = open(logF_name, 'w')
    logFile.write("=======================================================\n")

    net = None
    if 'NPFLD' in model_type:
        net = NPFLD(num_of_pts=pts_num, alpha=alpha)
    if 'CPFLD' in model_type:
        net = CPFLD(num_of_pts=pts_num, alpha=alpha)
    if 'BASE' in model_type:
        net = BASE(num_of_pts=pts_num)
    if 'MSBASE' in model_type:
        net = MSBASE(num_of_pts=pts_num)
    if 'M1BASE' in model_type:
        net = M1BASE(num_of_pts=pts_num)
    net.initialize(mx.init.Normal(sigma=0.001), ctx=devices, force_reinit=True)

    net.hybridize()
    if args.pretrain_param is not None:
        net.load_parameters(args.pretrain_param)

    huber_loss = mx.gluon.loss.HuberLoss(rho=5)
    mse_loss   = mx.gluon.loss.L2Loss()
    lmks_metric  = mx.metric.MAE()
    angs_metric  = mx.metric.MAE()


    lr_epoch   = []
    train_iter = mx.io.ImageRecordIter(
        path_imgrec=train_data_file, 
        data_shape=(3, image_size, image_size), 
        batch_size=batch_size,
        label_width=205,
        shuffle = True,
        shuffle_chunk_size = 1024,
        seed = 1234, 
        prefetch_buffer = 10, 
        preprocess_threads = 16
    )

    valid_iter = mx.io.ImageRecordIter(
        path_imgrec=valid_data_file, 
        data_shape=(3, image_size, image_size), 
        batch_size=50,
        label_width=205,
        shuffle = False,
        preprocess_threads = 16,
    )

       
    ## trainning
    trainer = mx.gluon.Trainer(
        params=net.collect_params(),
        #optimizer='sgd',
        #optimizer_params={'learning_rate': base_lr, 'momentum': 0.9, 'wd': 5e-5}
        optimizer='adam',
        optimizer_params={'learning_rate': base_lr}
    )

    for epoch in range(0, epoches):
        # reset training learning rate
        if (epoch+1) in lr_epoch:
            idx = 0
            for i in range(0, len(lr_epoch)):
                idx = i
                if (epoch+1) == lr_epoch[i]:
                    break
            lr = base_lr * math.pow(0.1, idx+1)
            trainer.set_learning_rate(lr)
        # reset data iterator
        train_iter.reset()
        valid_iter.reset()
        batch_idx = 0
        for batch in train_iter:
            batch_idx += 1
            batch_size = batch.data[0].shape[0]
            data   = batch.data[0]
            data   = preprocess(data)
            labels = batch.label[0]
            lmks = labels[:, 0:98*2] * image_size
            cate = labels[:, 2*98+1:2*98+6]
            angs = labels[:, -3:] * np.pi / 180.0

            cat_ratios = nd.mean(cate, axis=0)
            cat_ratios = (cat_ratios > 0.0)  * (1.0 / (cat_ratios+0.00001))
            cate       = cate * cat_ratios
            cate       = nd.sum(cate, axis=1) 
            cate       = (cate <= 0.0001) * 1 + cate

            data_list = mx.gluon.utils.split_and_load(data, ctx_list=devices, even_split=False)
            lmks_list = mx.gluon.utils.split_and_load(lmks, ctx_list=devices, even_split=False)
            angs_list = mx.gluon.utils.split_and_load(angs, ctx_list=devices, even_split=False)
            cate_list = mx.gluon.utils.split_and_load(cate, ctx_list=devices, even_split=False)
            loss_list = []

            with mx.autograd.record():
                for data, lmks, angs, cate in zip(data_list, lmks_list, angs_list, cate_list):
                    lmks_regs = net(data)
                    lmks_regs = nd.Flatten(lmks_regs)

                    lmks_loss = nd.square(lmks_regs - lmks)
                    lmks_loss = nd.sum(lmks_loss, axis=1)

                    #angs_loss = 1 - mx.nd.cos((angs_regs - angs)) 
                    #angs_loss = mx.nd.sum(angs_loss, axis=1)

                    loss = lmks_loss

                    #if with_angle:
                    #    loss = loss * angs_loss

                    if with_category:
                        loss = loss * cate

                    loss_list.append(loss)

                    lmks_metric.update(lmks, lmks_regs)
            for loss in loss_list:
                loss.backward()
            trainer.step(batch_size=batch_size, ignore_stale_grad=True)

            batch_loss = sum([l.sum().asscalar() for l in loss_list]) / batch_size
            #print('epoch:{}--{}'.format(epoch, batch_idx), 'loss={}'.format(batch_loss))
        # print infos, and save models after epoch
        lmks_name, lmks_mae = lmks_metric.get()
        angs_name, angs_mae = angs_metric.get()

        print('After epoch {}: {} = {}, {}={}, learning-rate={}, model_type---{}'.format(epoch + 1, lmks_name, lmks_mae, angs_name, angs_mae, trainer.learning_rate, model_type)) 
        net.export(os.path.join(output_dir, 'lmks_detector'), epoch=epoch+1)
        #net.save_parameters(os.path.join(output_dir, 'lmks_detector_{}.params'.format(epoch+1)))
        lmks_metric.reset()
        angs_metric.reset()
       
        # validate model in test data
        NME = 0.0
        FR  = 0.0
        NUM = 0
        for batch in valid_iter:
            data = batch.data[0]
            data = preprocess(data)
            labels = batch.label[0]
            lmks = labels[:, 0:98*2] * image_size
            angs = labels[:, -3:] * np.pi / 180.0
            data = data.as_in_context(devices[0])
            lmks = lmks.as_in_context(devices[0])
            angs = angs.as_in_context(devices[0])
            regs = net(data)
            regs = nd.Flatten(regs)
            batch_size = data.shape[0]
            NUM += batch_size
            regs = regs.asnumpy()
            lmks = lmks.asnumpy()
            for i in range(0, batch_size):
                ne = 0.0
                for j in range(0, 98):
                    e = (regs[i, j*2 + 0] - lmks[i, j*2 + 0]) * (regs[i, j*2 + 0] - lmks[i, j*2 + 0]) + \
                        (regs[i, j*2 + 1] - lmks[i, j*2 + 1]) * (regs[i, j*2 + 1] - lmks[i, j*2 + 1])
                    e = np.sqrt(e)
                    ne += e
                inter_occular=(lmks[i, 2*60 + 0] - lmks[i, 2*72 + 0]) * (lmks[i, 2*60 + 0] - lmks[i, 2*72 + 0]) +\
                              (lmks[i, 2*60 + 1] - lmks[i, 2*72 + 1]) * (lmks[i, 2*60 + 1] - lmks[i, 2*72 + 1])
                inter_occular = np.sqrt(inter_occular)
                ne = ne / (inter_occular * 98.0)
                NME += ne
                if ne > 0.1:
                    FR += 1.0
        NME /= NUM  
        FR  /= NUM

        print('Validaton: {} = {}, {} = {}'.format('NME', NME, 'FR', FR))

        val_log = 'epoch-{}, Validaton: {} = {}, {} = {}'.format(epoch, 'NME', NME, 'FR', FR)
        logFile.write(val_log + "\n")
        logFile.flush()

    if logFile is not None:
        logFile.close()
