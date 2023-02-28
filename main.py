# -*- coding: utf-8 -*-
import argparse
import numpy as np
from models.MLPAE import *
from models.GVAE import *
from models.AGVAE import *
from models.STAGED import *
from utils.setting import *
from utils.evaluate import *
from utils.load_data import *
from utils.metric import *
from train import train
from test import test, test_after_train
import visdom
import random
import datetime


class Main():
    def __init__(self, data_config, model_config, train_config, env_config, viz=True):
        self.data_config = data_config
        self.model_config = model_config
        self.train_config = train_config
        self.env_config = env_config


        set_device(env_config['device'])
        self.deice = get_device()

        self.train_loader, self.val_loader = load_train_val_data(use_dataset=data_config['use_dataset'],
                                                                 slide_win=data_config['slide_win'],
                                                                 slide_stride=data_config['slide_stride'],
                                                                 val_ratio=data_config['val_ratio'],
                                                                 batch_size=data_config['batch_size'],
                                                                 temporal_length=data_config['temporal_length'])

        self.test_loader = load_test_data(use_dataset=data_config['use_dataset'],
                                          slide_win=data_config['slide_win'],
                                          slide_stride=data_config['slide_stride'],
                                          batch_size=data_config['batch_size'],
                                          temporal_length=data_config['temporal_length'])

        if model_config['model_name'] == 'MLPAE':
            self.model = MLPAE(
                x_features=model_config['input_dim'],
                z_features=model_config['embedding_dim'],
                enc_layer=model_config['encoder_layer_num'],
                dec_layer=model_config['decoder_layer_num']
            )
        elif model_config['model_name'] == 'GVAE':
            self.model = GVAD(
                x_features=model_config['input_dim'],
                z_features=model_config['embedding_dim'],
                enc_layer=model_config['encoder_layer_num'],
                dec_layer=model_config['decoder_layer_num'])
        elif model_config['model_name'] == 'AGVAE':
            self.model = AGVAE(
                x_features=model_config['input_dim'],
                z_features=model_config['embedding_dim'],
                enc_layer=model_config['encoder_layer_num'],
                dec_layer=model_config['decoder_layer_num'])
        elif model_config['model_name'] == 'STAGE':
            self.model = STAGED(
                x_features=model_config['input_dim'],
                z_features=model_config['embedding_dim'],
                enc_layer=model_config['encoder_layer_num'],
                dec_layer=model_config['decoder_layer_num'])
        else:
            raise TypeError('No such model type!')

        if viz:
            self.viz = visdom.Visdom(env=data_config['use_dataset'] + train_config['comment'], port=6006)
        else:
            self.viz = None

    def run(self):
        if len(self.env_config['load_model_path']) > 0:
            model_save_path = self.env_config['load_model_path']
        else:
            model_save_path = get_save_path(save_path=env_config['save_path'], comments=train_config['comment'])[0]
            self.train_log = train(
                model=self.model,
                save_path=model_save_path,
                config=self.train_config,
                data_config=self.data_config,
                train_loader=self.train_loader,
                val_loader=self.val_loader,
                test_loader=self.test_loader,
                viz=self.viz,
                viz_test=True
            )

        self.model.load_state_dict(torch.load(model_save_path))
        best_model = self.model.to(self.deice)
        val_error_list, val_label_list = test_after_train(best_model, self.val_loader)
        test_error_list, test_label_list = test_after_train(best_model, self.test_loader)


        normalize_test_error = normalize_error(test_error_list, val_error_list, method='std', dataset=self.data_config['use_dataset'])

        test_label = np.array(test_label_list)
        if self.env_config['report'] == 'best':
            info = get_best_performance_data(normalize_test_error, test_label, topk=self.train_config['topk'])
        else:
            info = get_val_performance_data(normalize_test_error, test_label, topk=self.train_config['topk'])
        print(f'F1 score: {info[0]}')
        print(f'precision: {info[1]}')
        print(f'recall: {info[2]}\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # data_config
    parser.add_argument('-dataset', help='swat / wadi', type=str, default='swat')
    parser.add_argument('-slide_win', help='slide_win', type=int, default=32)
    parser.add_argument('-slide_stride', help='slide_stride', type=int, default=32)
    parser.add_argument('-val_ratio', help='val ratio', type=float, default=0.2)
    parser.add_argument('-batch', help='batch size', type=int, default=80)
    parser.add_argument('-temporal_length', help='temporal_length', type=int, default=4)  # if not STAGED, the temporal_length should be 1.

    # model_config
    parser.add_argument('-model_name', help='MLPAE / GVAE / AGVAE / STAGE', type=str, default='STAGE')
    parser.add_argument('-embedding_dim', help='embedding_dimension', type=int, default=64)
    parser.add_argument('-encoder_layer_num', help='encoder_layer_num', type=int, default=3)
    parser.add_argument('-decoder_layer_num', help='decoder_layer_num', type=int, default=3)

    # train_config
    parser.add_argument('-epoch', help='train epoch', type=int, default=10)
    parser.add_argument('-learning_rate', help='learning_rate', type=float, default=0.001)
    # (0-1) for reconstruction and prediction error
    parser.add_argument('-alpha', help='alpha', type=float, default=0.9)
    parser.add_argument('-decay', help='adam_weight_decay', type=float, default=5e-4)
    parser.add_argument('-early_stop_win', help='early_stop_win', type=int, default=10)
    parser.add_argument('-topk', help='topk_channel_to_evaluate_model', type=int, default=5)
    # encoder_layer  decoder_layer  temporal_length  hidden_dim
    parser.add_argument('-comment', help='experiment comment', type=str, default='3e_3d_4t_64')


    # env_config
    parser.add_argument('-device', help='cuda / cpu', type=str, default='cuda:1')
    parser.add_argument('-report', help='best / val', type=str, default='best')
    parser.add_argument('-load_model_path', help='trained model path', type=str, default='')
    parser.add_argument('-random_seed', help='random_seed', type=int, default=0)

    args = parser.parse_args()
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(args.random_seed)

    data_config = {
        'use_dataset': args.dataset,
        'slide_win': args.slide_win,
        'slide_stride': args.slide_stride,
        'val_ratio': args.val_ratio,
        'batch_size': args.batch,
        'temporal_length': args.temporal_length
    }

    model_config = {
        'model_name': args.model_name,
        'input_dim': args.slide_win,
        'embedding_dim': args.embedding_dim,
        'encoder_layer_num': args.encoder_layer_num,
        'decoder_layer_num': args.decoder_layer_num,
    }

    train_config = {
        'epoch': args.epoch,
        'comment': args.comment,
        'seed': args.random_seed,
        'lr': args.learning_rate,
        'decay': args.decay,
        'early_stop_win': args.early_stop_win,
        'topk': args.topk,
        'alpha': args.alpha,
    }

    env_config = {
        'save_path': args.dataset,
        'report': args.report,
        'device': args.device,
        'load_model_path': args.load_model_path
    }

    # The anomalous data are labelled with 1, while the normal data are labelled with 0.
    main = Main(data_config, model_config, train_config, env_config)
    main.run()
