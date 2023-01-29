import os
import argparse
from pytorch_nn import Lipread2, Lipread3
from datasetloading import dataloaders
from utilities import get_save_folder, logging_setup
from Model_training_blocks import FullTrainer


def load_args(default_config=None):
    parser = argparse.ArgumentParser(description='Lipreading model training')

    # -- Model Config
    parser.add_argument('--model-version', type=str, default='Lipread3', choices=['Lipread2', 'Lipread3'],
                        help='Choose between Lipread2 or Lipread3 to train')
    parser.add_argument('--num-classes', type=int, default=500, help='Number of classes')
    # -- Directories and Paths
    parser.add_argument('--model-path', type=str, default=None, help='Pretrained Model_Weights/Checkpoint path')
    parser.add_argument('--wordlist-file', type=str, default='./wordlist.txt', help='Path to wordlist text file')
    parser.add_argument('--data-direc', default=None, help='Preprocessed data directory')
    parser.add_argument('--lrw-direc', default=None, help='LRW Data Directory')
    # -- Training
    parser.add_argument('--batch-size', type=int, default=64, help='Mini-batch size')
    parser.add_argument('--workers', default=8, type=int, help='Number of data loading workers')
    parser.add_argument('--lr', default=0.001, type=float, help='Initial learning rate')
    parser.add_argument('--epochs', default=80, type=int, help='Number of epochs')
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adam', 'sgd', 'adamw'])
    parser.add_argument('--model-weights-only', default=True, help='Whether to load full checkpoint or just weights')
    # -- Mixup
    parser.add_argument('--use-mixup', default=True, help='Whether to use mixup when training')
    # -- WandB
    parser.add_argument('--use-wandb', default=False, help='Whether to use Weights and Biases when training (Will need to sign in)')
    parser.add_argument('--wandb-project-name', default=None, help='Project name to save wandb data. Defaults to model-version')
    parser.add_argument('--wandb-trainlog-interval', type=int, default=100,
                        help='How often to get training acc and loss in epoch (0 means only val acc and loss)')
    # -- Other
    parser.add_argument('--logging-direc', type=str, default='./train_logs',
                        help='path to the directory in which to save the log file')
    parser.add_argument('--test-mode', default=False, help='Get accuracy and loss on Test Set and TensorBoard insights')

    args = parser.parse_args()
    return args


args = load_args()


def set_up_dataloader():
    assert os.path.isdir(args.data_direc), "Preprocessed video directory does not exist. Path input {}".format(args.data_direc)
    if args.lrw_direc is not None:
        assert os.path.isdir(args.lrw_direc), "LipReadinginTheWild direc does not exist. Path input {}".format(args.lrw_direc)
    dataset = dataloaders(data_dir=args.data_direc, label_fp=args.wordlist_file, batch_size=args.batch_size,
                          lrw_direc=args.lrw_direc, workers=args.workers)
    return dataset


def main():

    save_loc = get_save_folder(args)
    print("Model and Log will be saved in: {}".format(save_loc))
    logger = logging_setup(args, save_loc)

    if args.model_version.lower() == 'lipread3':
        model = Lipread3(args.num_classes)
        logger.info("Lipread3 model loaded")
    elif args.model_version.lower() == 'lipread2':
        model = Lipread2(args.num_classes)
        logger.info("Lipread2 model loaded")
    else:
        logger.error("Please Specify 'Lipread2' or 'Lipread3'. Input was: {}".format(args.model_version))
        return

    model.cuda()

    dataset = set_up_dataloader()
    if args.test_mode:
        epochs = 0
    else:
        epochs = args.epochs

    model_trainer = FullTrainer(model, dataset, epochs, save_dir=save_loc, state_path=args.model_path,
                                model_weights_only=args.model_weights_only, lr=args.lr, optim=args.optimizer, logger=logger)

    if args.use_wandb:
        if args.wandb_project_name is None:
            model_trainer.wandb_activation(project_name=args.model_version, log_interval=args.wandb_trainlog_interval)
        else:
            model_trainer.wandb_activation(project_name=args.wandb_project_name, log_interval=args.wandb_trainlog_interval)

    if not args.test_mode:
        model_trainer.initialize_training(mixup=args.use_mixup)

if __name__ == '__main__':
    main()
