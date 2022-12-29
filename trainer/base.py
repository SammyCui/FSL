import abc
import json
import torch
import os
import torch.nn.functional as F
from data.task_loader import task_loader
from utils.logger import Logger
from utils.meters import AverageMeter, StatsMeter
from trainer.helpers import get_model_optimizer, partition_task, sort_batch
from utils.metric import accuracy


class BaseTrainer(metaclass=abc.ABCMeta):
    def __init__(self, args):
        self.logger = Logger(args)
        self.args = args
        self.train_step = 0
        self.train_epoch = args.start_epoch
        self.start_epoch = args.start_epoch
        self.max_steps = args.episodes_per_epoch * args.max_epoch
        self.max_epoch = args.max_epoch
        self.device = args.device

        self.train_time, self.forward_tm, self.backward_tm, self.optimize_tm = (AverageMeter(),) * 4

        self.result_log = {'max_val_acc': 0,
                           'max_val_acc_ci': 0,
                           'max_val_acc_epoch': 0}

        self.model, self.optimizer, self.lr_scheduler = get_model_optimizer(args)
        self.model.to(self.device)
        self.train_dataloader, self.val_dataloader, self.test_dataloader = task_loader(args)
        self.support_idx_train, self.query_idx_train = partition_task(self.args.n_ways_train, self.args.n_shots_train,
                                                                      self.args.n_queries_train)
        self.support_idx_test, self.query_idx_test = partition_task(self.args.n_ways_test, self.args.n_shots_test,
                                                                    self.args.n_queries_test)
        self.model.support_idx_train, self.model.support_idx_test, self.model.query_idx_train, self.model.query_idx_test = \
            self.support_idx_train, self.support_idx_test, self.query_idx_train, self.query_idx_test

        self.best_model_params = None

    @abc.abstractmethod
    def train(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def _validate(self):
        """

        :return: val_acc, val_loss, va_ci (validation accuracy confidence interval)
        """
        raise NotImplementedError()

    def validate(self):
        if self.train_epoch % self.args.val_interval == 0:
            val_acc, val_loss, va_ci = self._validate()

            if val_acc >= self.result_log['max_val_acc']:
                self.result_log['max_val_acc'] = val_acc
                self.result_log['max_val_acc_ci'] = va_ci
                self.result_log['max_val_acc_epoch'] = self.train_epoch
                self.best_model_params = self.model.state_dict()
                if self.args.save:
                    self.save_model('checkpoint')

            return val_acc, val_loss, va_ci

    def save_model(self, name):
        assert self.model is not None, "No models to be saved."
        checkpoint = {'models': self.model.state_dict(),
                      'optimizer': self.optimizer.state_dict()}
        if self.lr_scheduler:
            checkpoint['lr_scheduler'] = self.lr_scheduler.state_dict()
        torch.save(checkpoint, os.path.join(self.args.result_dir, name + '.pt'))

    def logging(self,
                train_loss,
                train_acc,
                val_loss,
                val_acc,
                val_ci):
        assert self.optimizer is not None, "Has not initialize optimizer yet."

        if self.train_epoch % self.args.val_interval == 0:
            print('epoch {}/{}, **Train** loss={:.4f} acc={:.4f} | ' \
                  '**Val** loss={:.4f} acc={:.4f}+{:.4f}, lr={:.4g}'
                  .format(self.train_epoch,
                          self.max_epoch,
                          train_loss, train_acc,
                          val_loss, val_acc, val_ci,
                          self.optimizer.param_groups[0]['lr']))
            self.logger.add_scalar('train_loss', train_loss, self.train_epoch)
            self.logger.add_scalar('train_acc', train_acc, self.train_epoch)
            self.logger.add_scalar('val_loss', val_loss, self.train_epoch)
            self.logger.add_scalar('val_acc', val_acc, self.train_epoch)

    def finish(self):
        self.logger.save_logger()
        print("==>", 'Training Statistics')

        for k, v in self.result_log.items():
            print(k, ': ', '{:.3f}'.format(v))

        print(
            'forward_timer  (avg): {:.2f} sec  \n' \
            'backward_timer (avg): {:.2f} sec, \n' \
            'optim_timer (avg): {:.2f} sec \n' \
            'epoch_timer (avg): {:.2f} hrs \n' \
            'total time to converge: {:.2f} hrs' \
                .format(
                    self.forward_tm.avg, self.backward_tm.avg,
                    self.optimize_tm.avg, self.train_time.avg / 3600,
                    self.train_time.sum / 3600
                )
        )
        with open(os.path.join(self.args.result_dir, 'results.txt'), 'w') as f:
            f.write('best epoch {}, best val acc={:.4f} + {:.4f}\n'.format(
                self.result_log['max_val_acc_epoch'],
                self.result_log['max_val_acc'],
                self.result_log['max_val_acc_ci']))
            f.write('Test acc={:.4f} + {:.4f}\n'.format(
                self.result_log['test_acc'],
                self.result_log['test_acc_ci']))
            f.write('Total time to converge: {:.3f} hrs, per epoch: {:.3f} hrs'
                    .format(self.train_time.sum / 3600, self.train_time.avg / 3600))

        self.logger.close()

    def test(self):
        print('==> Testing start')
        self.model.load_state_dict(self.best_model_params)
        self.model.eval()

        test_loss, test_acc = StatsMeter(), StatsMeter()
        with torch.no_grad():
            for batch in self.val_dataloader:
                data, label = sort_batch(batch)
                data, label = data.to(self.device), label.to(self.device)
                logits = self.model(data)

                query_label = label[self.query_idx_test]
                loss = F.cross_entropy(logits, query_label)
                acc = accuracy(logits, query_label)[0]

                test_loss.update(loss.item())
                test_acc.update(acc[0].item())

            test_ci = test_acc.compute_ci()

        self.result_log['test_acc'] = test_acc.avg
        self.result_log['test_loss'] = test_loss.avg
        self.result_log['test_acc_ci'] = test_ci

    def __str__(self):
        return "{}({})".format(
            self.__class__.__name__,
            self.model.__class__.__name__
        )
