from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CyclicLR, StepLR
from base_models_NN import Encoder, Predictor, Discriminator
from utils import evaluate_regression_results
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the Quantile Loss function
def quantile_loss(y_pred, y_true, tau):

    error = y_true - y_pred
    loss = torch.max((tau - 1) * error, tau * error).to(DEVICE)
    
    return loss

def modified_quantile_loss(y_pred, y_true, tau = 0.5, alpha = 2.0):

    error = y_true - y_pred
    loss = torch.where(error < 0, alpha * (tau - 1) * error, tau * error).to(DEVICE)

    return torch.mean(loss)

# Training settings
class Solver(object):
    def __init__(self, num_predictors, num_shared_feature, learning_rate=0.01, dropout_ratio = 0.2, optimizer='momentum', weight_decay=0.001):

        self.num_predictors = num_predictors
        self.num_shared_feature = num_shared_feature
        self.dropout_ratio = dropout_ratio
        self.optim = optimizer
        self.weight_decay = weight_decay

        self.E = Encoder(self.num_predictors, self.num_shared_feature, self.dropout_ratio).to(DEVICE)
        self.P = Predictor(self.num_shared_feature, self.dropout_ratio).to(DEVICE)
        self.D = Discriminator(self.num_shared_feature, self.dropout_ratio).to(DEVICE)

        self.prediction_criteria = torch.nn.MSELoss(reduction='none').to(DEVICE)#torch.nn.MSELoss().to(DEVICE); torch.nn.L1Loss().to(DEVICE)
        self.classification_criteria = torch.nn.CrossEntropyLoss(reduction='none').to(DEVICE)

        self.set_optimizer(which_opt=optimizer, lr=learning_rate, weight_decay = weight_decay)
        self.lr = learning_rate

    def set_optimizer(self, which_opt, lr, weight_decay=0.001, momentum=0.9):
        if which_opt == 'momentum':
            self.opt_e = optim.SGD(self.E.parameters(),
                                   lr=lr, weight_decay=weight_decay,
                                   momentum=momentum)

            self.opt_p = optim.SGD(self.P.parameters(),
                                    lr=lr, weight_decay=weight_decay,
                                    momentum=momentum)

            self.opt_d = optim.SGD(self.D.parameters(),
                                    lr=lr, weight_decay=weight_decay,
                                    momentum=momentum)

        if which_opt == 'adam':
            self.opt_e = optim.Adam(self.E.parameters(),
                                    lr=lr, weight_decay=weight_decay)

            self.opt_p = optim.Adam(self.P.parameters(),
                                     lr=lr, weight_decay=weight_decay)

            self.opt_d = optim.Adam(self.D.parameters(),
                                     lr=lr, weight_decay=weight_decay)

        if which_opt == 'Adagrad':
            self.opt_e = optim.Adagrad(self.E.parameters(),
                                    lr=lr, weight_decay=weight_decay)

            self.opt_p = optim.Adagrad(self.P.parameters(),
                                     lr=lr, weight_decay=weight_decay)

            self.opt_d = optim.Adagrad(self.D.parameters(),
                                     lr=lr, weight_decay=weight_decay)

        self.scheduler_e = StepLR(self.opt_e, 50, 0.5) #CyclicLR(self.opt_e, base_lr=lr, max_lr=max_lr, mode='triangular')
        self.scheduler_p = StepLR(self.opt_p, 50, 0.5) #CyclicLR(self.opt_p, base_lr=lr, max_lr=max_lr, mode='triangular')
        self.scheduler_d = StepLR(self.opt_d, 50, 0.5) #CyclicLR(self.opt_d, base_lr=lr, max_lr=max_lr, mode='triangular')


    def reset_grad(self):
        self.opt_e.zero_grad()
        self.opt_p.zero_grad()
        self.opt_d.zero_grad()

    def train(self, dataloader_src, train_loader_src_all, dataloader_tar, dataloader_test, train_loader_loc, epoch, nepoch):

        weight_10 = 1.0
        weight_50 = 1.0
        weight_90 = 1.0

        for epoch in range(nepoch):

            self.E.train()
            self.P.train()
            self.D.train()

            loss_p_src_m = 0
            loss_d_src_m = 0
            loss_d_tar_m = 0

            for batch_idx, (data_src, data_tar) in enumerate(zip(dataloader_src, dataloader_tar)):

                x_src, y_src = data_src
                x_tar, y_tar = data_tar

                dist_src = x_src[:,-1]
                dist_src = torch.unsqueeze(dist_src, 1)


                dist_src = dist_src / max(dist_src)
                dist_src_inv = 1.0 / dist_src
                #dist_src_inv = dist_src
                dist_src_inv = dist_src_inv.to(DEVICE)
                #dist_src_inv = 1
                x_src = x_src[:,0:-1]

                size_src = x_src.shape[0]
                size_tar = x_tar.shape[0]
                x_src, y_src, x_tar = x_src.to(DEVICE), y_src.to(DEVICE), x_tar.to(DEVICE)

                # learning process
                p = float(epoch / nepoch)
                alpha = (2. / (1. + np.exp(-p)) - 1)

                ## Step 1: pretrain on source domain
                self.reset_grad()
                feat_s, _, _, _, _, _ = self.E(x_src)
                feat_t, _, _, _, _, _ = self.E(x_tar)

                y_pred_s = self.P(feat_s)
                alpha = 1
                domain_pred_s = self.D(feat_s, alpha)
                domain_pred_t = self.D(feat_t, alpha)

                # Source Domain - 0, target domain - 1
                domain_label_s = torch.zeros(size_src)
                domain_label_s = domain_label_s.long()
                domain_label_s = domain_label_s.to(DEVICE)

                domain_label_t = torch.ones(size_tar)
                domain_label_t = domain_label_t.long()
                domain_label_t = domain_label_t.to(DEVICE)

                # Calculate the loss
                loss_10 = quantile_loss(y_pred_s, y_src, 0.1)
                loss_50 = quantile_loss(y_pred_s, y_src, 0.5)
                loss_90 = quantile_loss(y_pred_s, y_src, 0.9)

                loss_10_w = torch.mul(loss_10, dist_src_inv)
                loss_10_w = loss_10_w[loss_10_w != 0]

                loss_50_w = torch.mul(loss_50, dist_src_inv)
                loss_50_w = loss_50_w[loss_50_w != 0]

                loss_90_w = torch.mul(loss_90, dist_src_inv)
                loss_90_w = loss_90_w[loss_90_w != 0]

                loss_10 = torch.mean(loss_10_w)
                loss_50 = torch.mean(loss_50_w)
                loss_90 = torch.mean(loss_90_w)

                err_s_pred = 1/3 * (weight_10 * loss_10 + weight_90 * loss_90 + weight_50 * loss_50)


                err_s_domain = self.classification_criteria(domain_pred_s, domain_label_s)
                err_s_domain = torch.mean((torch.mul(err_s_domain, dist_src_inv)))

                err_t_domain = self.classification_criteria(domain_pred_t, domain_label_t)
                err_t_domain = torch.mean(err_t_domain)

                loss = err_s_pred + 1/2 * (err_s_domain + err_t_domain) # err_s_pred + alpha *

                loss.backward()

                self.opt_e.step()
                self.opt_p.step()
                self.opt_d.step()

                self.reset_grad()

            if epoch % 100 == 0:
                print("Epoch = ", epoch)
                print("prediction loss = ", err_s_pred.item())
    
            if epoch == 200:
                RMSE, R2, MAPE, r2, MAE, y_src, y_src_pred = self.test(train_loader_loc, False)
                error = (y_src - y_src_pred)
                error_pos = error[error > 0]
                error_neg = error[error < 0]
                ratio_1 = np.exp(len(error_pos) / error.shape[0])
                ratio_2 = np.exp(1 - len(error_pos) / error.shape[0])
                print("Evaluation on local county-level data RMSE, R2, MAPE = ", RMSE, R2, MAPE)

                weight_90 = np.exp(ratio_1 ** 2) / (np.exp(ratio_1 ** 2) + np.exp(ratio_2 ** 2)) * 2
                #weight_90 = np.exp(ratio_1 ** 0.5) / (np.exp(ratio_1 ** 0.5) + np.exp(ratio_2 ** 0.5)) * 2
                weight_10 = 2 - weight_90
                    
                print("weight 10 = ", weight_10)
                print("weight_90 = ", weight_90)

            #Update the learning at the end of training
            self.scheduler_e.step()
            self.scheduler_p.step()
            self.scheduler_d.step()

        return loss_p_src_m / (batch_idx + 1), 0.0 / (batch_idx + 1), 0.0 / (batch_idx + 1)


    def test(self, dataloader, print_out = False):
        self.E.eval()
        self.P.eval()
        self.D.eval()

        for batch_idx, data in enumerate(dataloader):
            x, y = data
            x, y = x.to(DEVICE), y.to(DEVICE)

            feat, attn, x1, x2, x3, x4 = self.E(x)
            y_pred = self.P(feat)

            feat = feat.detach()
            #attn = attn.detach()
            x1 = x1.detach()
            x2 = x2.detach()
            x3 = x3.detach()
            x4 = x4.detach()

            y = y.detach()
            y_pred = y_pred.detach()

            y_pred[y_pred > 20] = 20
            y_pred[y_pred < 0] = 0

            error = y - y_pred

            y = y.cpu().numpy()
            y_pred = y_pred.cpu().numpy()

            RMSE, R2, MAPE, r2, MAE = evaluate_regression_results(y, y_pred)

        if print_out == True:
            print('Test RMSE = {:.4f}, R2 = {:.4f}, MARE = {:.4f}, r2 = {:.4f} \n'.format(RMSE, R2, MAPE, r2))

        return RMSE, R2, MAPE, r2, MAE, y, y_pred


    def predict(self, x, print_out = False):
        self.E.eval()
        self.P.eval()
        self.D.eval()

        # x is numpy not tensor, return is numpy
        xx = torch.tensor(x, dtype=torch.float32).to(DEVICE)

        with torch.no_grad():
            feat, attn = self.E(xx)
            y_pred = self.P(feat)
            y_pred = y_pred.detach()
            y_pred = y_pred.cpu().numpy()

        return y_pred

    def load_models(self, year, path):

        save_path_encoder       = path + "QDANN_E_corn_" + str(year) + ".pth"
        save_path_predictor     = path + "QDANN_P_corn_" + str(year) + ".pth"
        save_path_discriminator = path + "QDANN_D_corn_" + str(year) + ".pth"

        self.E.load_state_dict(torch.load(save_path_encoder, map_location=torch.device(DEVICE)))
        self.P.load_state_dict(torch.load(save_path_predictor, map_location=torch.device(DEVICE)))
        self.D.load_state_dict(torch.load(save_path_discriminator, map_location=torch.device(DEVICE)))
