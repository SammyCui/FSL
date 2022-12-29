from timeit import default_timer as timer
import os
import torch
import torchvision
import torch.nn.functional as F
from trainer.base import BaseTrainer
from utils.meters import AverageMeter, StatsMeter
from data.task_loader import task_loader
from trainer.helpers import sort_batch
from utils.metric import accuracy


class EpisodicTrainer(BaseTrainer):
    def __init__(self, args):

        super().__init__(args)

    def train(self):

        print('==> Training Start')
        epoch_t0 = timer()
        for epoch in range(self.start_epoch, self.max_epoch+1):

            self.model.train()

            train_loss, train_acc = AverageMeter(), AverageMeter()

            for batch in self.train_dataloader:

                self.train_step += 1
                data, label = sort_batch(batch)
                data, label = data.to(self.device), label.to(self.device)

                forward_t0 = timer()
                logits = self.model(data)
                forward_t1 = timer()

                query_label = label[self.query_idx_train]
                loss = F.cross_entropy(logits, query_label)
                acc  = accuracy(logits, query_label)[0]

                self.optimizer.zero_grad()
                backward_t0 = timer()
                loss.backward()
                backward_t1 = timer()

                optimizer_t0 = timer()
                self.optimizer.step()
                optimizer_t1 = timer()

                train_loss.update(loss.item())
                train_acc.update(acc[0].item())
                self.backward_tm.update(backward_t1-backward_t0)
                self.forward_tm.update(forward_t1-forward_t0)
                self.optimize_tm.update(optimizer_t1-optimizer_t0)

            if self.lr_scheduler:
                self.lr_scheduler.step()
            val_acc, val_loss, val_ci = self.validate()

            epoch_t1 = timer()
            self.train_time.update(epoch_t1-epoch_t0)
            epoch_t0 = epoch_t1

            self.logging(train_loss=train_loss.avg, train_acc=train_acc.avg,
                         val_loss=val_loss, val_acc=val_acc, val_ci=val_ci)
            self.train_epoch += 1

    def _validate(self):
        self.model.eval()
        val_loss, val_acc = StatsMeter(), StatsMeter()
        with torch.no_grad():
            for batch in self.val_dataloader:
                data, label = sort_batch(batch)
                data, label = data.to(self.device), label.to(self.device)
                logits = self.model(data)

                query_label = label[self.query_idx_test]
                loss = F.cross_entropy(logits, query_label)
                acc = accuracy(logits, query_label)[0]

                val_loss.update(loss.item())
                val_acc.update(acc[0].item())

            val_ci = val_acc.compute_ci()

        return val_acc.avg, val_loss.avg, val_ci




