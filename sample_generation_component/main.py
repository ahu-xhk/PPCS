import argparse
import datetime

import utils
import component.detector as detector

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    datasets = ['amazon', 'dblp', 'twitter', 'youtube', 'lj']

    parser.add_argument('--dataset', type=str, default='youtube')
    parser.add_argument('--root', type=str, default='datasets')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--train_size', type=int, default=100) 
    parser.add_argument('--k_ego_subG', type=int, default=3)

    # Model
    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--g_lr', type=float, default=1e-2)

    # Train
    parser.add_argument('--g_batch_size', type=int, default=32) 
    parser.add_argument('--epochs', type=int, default=100)


    args = parser.parse_args()
    utils.seed_all(args.seed)

    print('= ' * 20)
    now = datetime.datetime.now()
    print('##  Starting Time:', now.strftime("%Y-%m-%d %H:%M:%S"), flush=True)

    print(f"dataset [{args.dataset}]")
    seeds, com_indexs = utils.getseedsAndtruecom(args, args.dataset)
    dt = detector.Detector(args, seeds, com_indexs)
    dt.detect()

    print('## Finishing Time:', datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), flush=True)
    print('= ' * 20)

