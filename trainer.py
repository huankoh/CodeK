import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from numpy.random import randint
from random import sample, choices
from sklearn.metrics import roc_curve, auc, average_precision_score
from torch.optim.lr_scheduler import OneCycleLR

class Trainer:
    def __init__(self, model, data, epochs, batch_size, lr, weight_decay, device):
        self.device = device
        self.model = model
        self.epochs = epochs
        self.batch_size = batch_size
        self.trainX, self.trainY, self.testX, self.testY, self.hd0x_train, self.hd1x_train, \
        self.hd0x_test, self.hd1x_test, self.hd0y_train, self.hd1y_train, self.hd0y_test, self.hd1y_test = data
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.best_model = model
        self.scheduler = OneCycleLR(self.optimizer, max_lr=lr, epochs=epochs, steps_per_epoch=self.trainX.shape[0] // batch_size + 1)

    def train(self):
        print('Start training...')

        contrastive_train_X, contrastive_train_Y = \
            torch.tensor(self.trainX).to(self.device), torch.tensor(self.trainY).to(self.device)

        hd0x_train, hd1x_train, hd0y_train, hd1y_train = \
            torch.tensor(self.hd0x_train).to(self.device), torch.tensor(self.hd1x_train).to(self.device), \
            torch.tensor(self.hd0y_train).to(self.device), torch.tensor(self.hd1y_train).to(self.device)

        best_loss = 1e9
        best_auc = 0.0
        for i in range(self.epochs):
            self.model.train()

            train_loss = 0.0
            alpha = 1 - i / self.epochs

            # shuffle training data for contrastive branch
            shuffle_idx = torch.randperm(contrastive_train_X.shape[0]).to(self.device)
            con_train_X = contrastive_train_X[shuffle_idx].view(contrastive_train_X.size())
            con_train_Y = contrastive_train_Y[shuffle_idx].view(contrastive_train_Y.size())

            # shuffle training data for downstream branch
            shuffle_idx = torch.randperm(hd0x_train.shape[0]).to(self.device)
            hd0x_train = hd0x_train[shuffle_idx].view(hd0x_train.size())
            hd0y_train = hd0y_train[shuffle_idx].view(hd0y_train.size())
            shuffle_idx = torch.randperm(hd1x_train.shape[0]).to(self.device)
            hd1x_train = hd1x_train[shuffle_idx].view(hd1x_train.size())
            hd1y_train = hd1y_train[shuffle_idx].view(hd1y_train.size())

            # iterate over all training samples
            num_batches = con_train_X.shape[0] // self.batch_size + 1
            all_idx = torch.tensor(list(range(con_train_X.shape[0]))).to(self.device)

            for batch_idx in range(num_batches):

                self.optimizer.zero_grad()

                is_final_batch = (batch_idx == (num_batches - 1))
                if not is_final_batch:
                    idx = all_idx[batch_idx * self.batch_size: (batch_idx + 1) * self.batch_size]
                else:
                    idx = all_idx[batch_idx * self.batch_size:]

                cur_batch_size = len(idx)

                # batch data for contrastive branch: Random sampling  # TODO: improve this branch (task 1)
                batch_con_x = con_train_X[idx]  # (batch, inp_dim)
                batch_con_y = con_train_Y[idx].squeeze(-1)  # (batch,)

                # batch data for downstream branch: Balanced sampling
                idx = torch.tensor(choices(list(range(hd0x_train.shape[0])), k=self.batch_size//2)).to(self.device)
                batch_down_x0 = hd0x_train[idx]
                batch_down_y0 = hd0y_train[idx]
                idx = torch.tensor(choices(list(range(hd1x_train.shape[0])), k=self.batch_size//2)).to(self.device)
                batch_down_x1 = hd1x_train[idx]
                batch_down_y1 = hd1y_train[idx]
                batch_down_x = torch.cat((batch_down_x0, batch_down_x1), dim=0)
                batch_down_y = torch.cat((batch_down_y0, batch_down_y1), dim=0)

                # fit model
                loss = self.model(batch_con_x, batch_con_y, batch_down_x, batch_down_y, alpha=alpha)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                self.optimizer.step()
                self.scheduler.step()

                loss = loss.detach().cpu().numpy()

                if not is_final_batch:
                    train_loss += loss

            mean_loss = (train_loss * self.batch_size + loss * cur_batch_size) / con_train_X.shape[0]

            if mean_loss < best_loss:
                self.best_model = self.model

            print('Epoch:{} Loss:{:.8f}'.format(i, mean_loss), flush=True)

            if i % 10 == 0:
                model_fpr, model_tpr, model_auc = self.evaluate(self.best_model)
                print('Testing statistics: AUC:{:.4f}'.format(model_auc), flush=True)
                if model_auc > best_auc:
                    best_auc = model_auc

        print('Best testing statistics: AUC:{:.4f}'.format(best_auc), flush=True)


    def evaluate(self, model):
        model = model.eval()
        print('Start evaluation...')
        testX = torch.tensor(self.testX).to(self.device)
        testY_hat = model.predict(testX)
        model_fpr, model_tpr, model_thresh = roc_curve(self.testY, testY_hat.cpu().detach().numpy())
        model_auc = auc(model_fpr, model_tpr)
        print('AUC: {}'.format(model_auc))

        return model_fpr, model_tpr, model_auc

