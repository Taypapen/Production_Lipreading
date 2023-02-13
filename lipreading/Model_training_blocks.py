
import time
import shutil

from tqdm import tqdm
from mixup import mixup_data, mixup_criterion
from utilities import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb


def train_loop(model, dataloader, criterion, epoch, optimizer, logger, mixup=False, log_interval=None, mixup_alpha = 0.4,
               ):

    mixup_alpha = mixup_alpha

    model.train()
    running_loss = 0.
    running_corrects = 0.
    running_all = 0.

    for batch_idx, (input, lengths, labels) in enumerate(tqdm(dataloader)):
        if mixup:
            input, labels_a, labels_b, lam = mixup_data(input, labels, mixup_alpha)
            labels_a, labels_b = labels_a.cuda(), labels_b.cuda()
        else:
            labels = labels.cuda()

        optimizer.zero_grad()

        logits = model(input.cuda(), lengths=lengths)

        if mixup:
            loss_func = mixup_criterion(labels_a, labels_b, lam)
            loss = loss_func(criterion, logits)
        else:
            loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        # -- compute running performance
        _, predicted = torch.max(F.softmax(logits, dim=1).data, dim=1)
        running_loss += loss.item()*input.size(0)

        if mixup:
            running_corrects += lam * predicted.eq(labels_a.view_as(predicted)).sum().item() + (1 - lam) * predicted.eq(labels_b.view_as(predicted)).sum().item()
            corrects = lam * predicted.eq(labels_a.view_as(predicted)).sum().item() + (1 - lam) * predicted.eq(
                labels_b.view_as(predicted)).sum().item()
        else:
            running_corrects += torch.sum(predicted == labels.data)
            corrects = torch.sum(predicted == labels.data)
        running_all += input.size(0)
        if log_interval is not None:
            if batch_idx % log_interval == 0:
                wandb.log({"train_loss": loss, "train_acc": (corrects/input.size(0))})
    logger.info("Running Loss: {}, Running Corrects: {}, Running All: {}".format(running_loss, running_corrects, running_all))

    return model


def evaluate(model, dset_loader, criterion, logger, profiler=None):

    model.eval()

    running_loss = 0.
    running_corrects = 0.
    if profiler is not None:
        profiler.start()

    with torch.no_grad():
        for batch_idx, (input, lengths, labels) in enumerate(tqdm(dset_loader)):
            logits = model(input.cuda(), lengths=lengths)
            _, preds = torch.max(F.softmax(logits, dim=1).data, dim=1)
            running_corrects += preds.eq(labels.cuda().view_as(preds)).sum().item()

            loss = criterion(logits, labels.cuda())
            running_loss += loss.item() * input.size(0)

            if profiler is not None:
                profiler.step()

    logger.info('{} in total\tCR: {}'.format(len(dset_loader.dataset), running_corrects/len(dset_loader.dataset)))
    if profiler is not None:
        profiler.stop()
    return running_corrects/len(dset_loader.dataset), running_loss/len(dset_loader.dataset)


class FullTrainer(object):
    def __init__(self, model, dataloader, epochs, criterion=nn.CrossEntropyLoss().cuda(), save_dir='./', state_path=None,
                 model_weights_only=False, lr=0.001, optim='adam', logger=None):
        # Model/Training Params
        self.epoch = 0
        self.model = model
        self.dataset = dataloader
        self.epochs = epochs
        self.lr = lr
        self.criterion = criterion
        self.scheduler = CosineScheduler(lr, epochs)
        self.allow_size_mismatch = True
        if logger is None:
            self.logger = logging_setup(None, './')
        else:
            self.logger = logger

        # Weights and Biases init values
        self.use_wandb = False
        self.log_interval = None

        # Checkpoint Params
        self.checkpoint_filename = 'ckpt.pth.tar'
        self.best_fn = 'ckpt.best.pth.tar'
        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir, exist_ok=True)
        self.best_perf = 0

        if state_path is None:
            self._initialize_weights_randomly()
            self.optimizer = get_optimizer(optim, self.model.parameters(), lr=lr)

        else:
            self.load_checkpoint(optim, state_path, model_weights_only)

        if epochs == 0:
            self.profiler_activation()
            self.test_performance()

    def load_checkpoint(self, optim_type, state_path, model_only):
        checkpoint = torch.load(state_path)
        loaded_state_dict = checkpoint['model_state_dict']
        if self.allow_size_mismatch:
            new_state_dict = {}
            for k, v in checkpoint['model_state_dict'].items():
                if k in self.model.state_dict() and self.model.state_dict()[k].shape == v.shape:
                    new_state_dict[k] = v
            loaded_state_dict = new_state_dict
        self.model.load_state_dict(loaded_state_dict, strict=False)
        self.optimizer = get_optimizer(optim_type, self.model.parameters(), lr=self.lr)
        if not model_only:
            self.epoch = checkpoint['epoch_idx']
            self.best_perf = checkpoint['best_perf']
            try:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            except KeyError:
                self.logger.error("State_path is incompatible with optimizer: {}\nChoose a different checkpoint to "
                                  "load or specify model_weights_only=True".format(str(optim_type)))
            if self.epoch < self.epochs:
                self.scheduler.adjust_lr(self.optimizer, self.epoch - 1)
            else:
                self.logger.warning("State_path: {} current Epoch: {} is not less than Epochs: {}\nIncrease "
                                    "total Epochs or specify model_weights_only=True. Setting current epoch to 0..."
                                    .format(state_path, str(self.epoch), str(self.epochs)))
                self.epoch = 0

    def _initialize_weights_randomly(self):
        use_sqrt = True

        if use_sqrt:
            def f(n):
                return math.sqrt( 2.0/float(n) )
        else:
            def f(n):
                return 2.0/float(n)

        for m in self.model.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                n = np.prod( m.kernel_size ) * m.out_channels
                m.weight.data.normal_(0, f(n))
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

            elif isinstance(m, nn.Linear):
                n = float(m.weight.data[0].nelement())
                m.weight.data = m.weight.data.normal_(0, f(n))

    def initialize_training(self, mixup=False):
        while self.epoch < self.epochs:
            start_epoch = time.time()
            self.logger.info('-' * 10)
            self.logger.info('Epoch {}/{}'.format(self.epoch + 1, self.epochs))
            self.logger.info('Current learning rate: {}'.format(showLR(self.optimizer)))

            self.model = train_loop(self.model, self.dataset['train'], self.criterion, self.epoch, self.optimizer,
                                    self.logger, mixup=mixup, log_interval=self.log_interval)
            acc_avg_val, loss_avg_val = evaluate(self.model, self.dataset['val'], self.criterion, self.logger)
            self.logger.info('{} Epoch:\t{:2}\tLoss val: {:.4f}\tAcc val:{:.4f}, LR: {}'
                             .format('val', self.epoch, loss_avg_val, acc_avg_val, showLR(self.optimizer)))
            if self.use_wandb:
                wandb.log({"val_acc": acc_avg_val, "val_loss": loss_avg_val})
            save_dict = {
                'epoch_idx': self.epoch + 1,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict()
                }
            self.save_checkpoint(save_dict, acc_avg_val)
            self.scheduler.adjust_lr(self.optimizer, self.epoch)
            self.epoch += 1
            epoch_len = time.time() - start_epoch
            self.logger.info("Epoch len: {} Estimated Remaining: {} Min"
                             .format(str(epoch_len), str(((self.epochs - self.epoch) * epoch_len) / 60)))
        self.eval_best_performance()

    def eval_best_performance(self):
        best_filepath = os.path.join(self.save_dir, self.best_fn)
        assert os.path.isfile(best_filepath)
        checkpoint = torch.load(best_filepath)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        acc_avg_test, loss_avg_test = evaluate(self.model, self.dataset['test'], self.criterion, self.logger)
        self.logger.info('Test time performance of best epoch: {} (loss: {})'.format(acc_avg_test, loss_avg_test))

    def save_checkpoint(self, save_dict, current_score):
        checkpoint_fp = os.path.join(self.save_dir, self.checkpoint_filename)

        self.is_best = current_score > self.best_perf
        if self.is_best:
            self.best_perf = current_score
            best_filepath = os.path.join(self.save_dir, self.best_fn)
        save_dict['best_perf'] = self.best_perf

        torch.save(save_dict, checkpoint_fp)
        if self.is_best:
            shutil.copyfile(checkpoint_fp, best_filepath)

    def wandb_activation(self, project_name="Lipread_Training", log_interval=0):
        self.use_wandb = True
        wandb.init(project=project_name)
        wandb.watch(self.model)
        if log_interval == 0:
            self.log_interval = None
        else:
            self.log_interval = log_interval

    def profiler_activation(self):
        self.prof = torch.profiler.profile(
                schedule=torch.profiler.schedule(wait=2, warmup=2, active=3),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(os.path.join(self.save_dir, 'profile_log')),
                record_shapes=True,
                profile_memory=True,
                with_stack=True)

    def test_performance(self):
        acc_avg_val, loss_avg_val = evaluate(self.model, self.dataset['test'], self.criterion, self.logger, self.prof)
        self.logger.info('{} Loss val: {:.4f}\tAcc val:{:.4f}'.format('test', loss_avg_val, acc_avg_val))
        return self.prof

    def save_model_weights(self, save_location=self.save_dir):
        save_dict = {'model_state_dict': self.model.state_dict()}
        file_path = os.path.join(save_location, 'model_weights.tar')
        torch.save(save_dict, file_path)
        self.logger.info('Model Weights Saved in: {}'.format(file_path))