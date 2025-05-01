import argparse
from copy import deepcopy
import torch
import misc
import optimizers
from models import MLP, CNN, fit

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-num_epochs', type=int, default=30)
    parser.add_argument('-dataset', type=str, default='cifar')
    parser.add_argument('-num_train', type=int, default=50000)
    parser.add_argument('-num_val', type=int, default=2048)
    parser.add_argument('-lr_schedule', type=bool, default=True)
    parser.add_argument('-only_plot', type=bool, default=True)
    args = parser.parse_args()

    # Load dataset
    data = getattr(misc, f'load_{args.dataset}')(
        num_train=args.num_train,
        num_val=args.num_val
    )
    print(f'Loaded data partitions: Train={len(data[0])}, Val={len(data[1])}')

    # Optimizer tasks to run
    opt_tasks = [
        'sgd', 'sgd_momentum', 'sgd_nesterov', 'sgd_weight_decay', 'sgd_lrd',
        'rmsprop', 'adam', 'adam_l2', 'adamW', 'adam_lrd',
        'Radam', 'RadamW', 'Radam_lrd', 'nadam',
        'lookahead_sgd', 'lookahead_adam', 'gradnoise_adam', 'graddropout_adam'
    ]
    opt_losses, opt_val_losses, opt_labels = [], [], []

    def train_optimizer(opt):
        print(f'\nTraining {opt} for {args.num_epochs} epochs...')
        net = CNN() if args.dataset == 'cifar' else MLP()
        _, kwargs = misc.split_optim_dict(misc.optim_dict[opt])
        optimizer = misc.task_to_optimizer(opt)(net.parameters(), **kwargs)
        optimizer = misc.wrap_optimizer(opt, optimizer)
        return fit(net, data, optimizer, args.num_epochs, args.lr_schedule)

    for opt in opt_tasks:
        if args.only_plot:
            losses = misc.load_losses(args.dataset, opt)
            val_losses = misc.load_losses(args.dataset, f'{opt}_val')
            if losses is None:
                print(f"No cached losses for {opt}. Training now...")
                args.only_plot = False
                losses, val_losses = train_optimizer(opt)
                misc.save_losses(losses, args.dataset, opt)
                misc.save_losses(val_losses, args.dataset, f'{opt}_val')
                args.only_plot = True
        else:
            print(f"Training {opt}...")  # Debug print
            losses, val_losses = train_optimizer(opt)
            misc.save_losses(losses, args.dataset, opt)
            misc.save_losses(val_losses, args.dataset, f'{opt}_val')

        if losses is not None:
            opt_losses.append(losses)
            opt_val_losses.append(val_losses)
            label = misc.split_optim_dict(misc.optim_dict[opt])[0]
            opt_labels.append(label)
            print(f"Collected Label: {label}")  # Debug print

    # Plot results if any data exists
    if len(opt_losses) > 0:
        misc.plot_losses(...)
    else:
        print("No results to plot. Train models first or check cached files.")