#! /usr/bin/env python3

import torch
from model import FCNN, CNN, ResNet  # noqa
from sklearn.metrics import log_loss, roc_auc_score, accuracy_score


def train(model, loader, optimizer, loss_fn, device, verbose):
    model.train()
    model.to(device)

    total_loss = 0

    for index, (data, target) in enumerate(loader):
        data = data.to(device)
        target = target.to(device)

        optimizer.zero_grad()

        output = model(data)

        loss = loss_fn(output, target)

        total_loss += loss.item()

        loss.backward()

        optimizer.step()

        if verbose:
            print(f'\rBatch {index + 1}/{len(loader)}, agg loss: {total_loss}',
                  end='')

    print(f'\nTotal loss: {total_loss}')


def test(model, loader, device):
    model.eval()
    model.to(device)

    targets = []
    outputs = []

    with torch.no_grad():
        for index, (data, target) in enumerate(loader):
            targets.append(target)

            data = data.to(device)
            output = model(data)
            outputs.append(output)

        y_pred = torch.cat(outputs, dim=0).cpu().numpy()
        y_true = torch.cat(targets, dim=0).cpu().numpy()

        print('\n' + '-' * 30)
        print('Results')
        print(f'Accuracy: {accuracy_score(y_true, y_pred > 0.5):.4f}')
        print(f'AUC: {roc_auc_score(y_true, y_pred):.4f}')
        print(f'Log loss: {log_loss(y_true, y_pred):.4f}')


def test_train_functions(train_dataset, val_dataset, test_dataset):
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=64,
                                               shuffle=True,
                                               num_workers=0,
                                               )

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=64,
                                             shuffle=False,
                                             num_workers=0,
                                             )

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=64,
                                              shuffle=False,
                                              num_workers=0,
                                              )

    device = 'cuda:0'

    model = FCNN().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = torch.nn.BCELoss()

    for i in range(5):
        print('\n' + '=' * 30)
        print(f'Training epoch {i + 1}')
        train(model=model,
              loader=train_loader,
              optimizer=optimizer,
              loss_fn=loss_fn,
              device=device,
              )

        test(model=model,
             loader=val_loader,
             device=device,
             )

    print('\n\n\n' + '+' * 30)
    print('Test set results')
    test(model=model,
         loader=test_loader,
         device=device,
         )


if __name__ == '__main__':
    from data_transform import CSGODataset, transform_data
    import sys
    import argparse
    import warnings
    import random
    warnings.filterwarnings('ignore')

    parser = argparse.ArgumentParser()

    parser.add_argument('--n-epochs',
                        type=int,
                        default=10,
                        )

    parser.add_argument('--batch-size',
                        type=int,
                        default=32,
                        )

    parser.add_argument('--model-type',
                        type=str,
                        default='fc',
                        )

    parser.add_argument('--hidden-sizes',
                        type=lambda s: [int(item) for item in s.split(',')],
                        default=[200, 100, 50],
                        )

    parser.add_argument('--cnn-options',
                        type=lambda x: tuple(tuple(int(item)
                                                   for item in s.split(','))
                                             for s in x.split('|')),
                        default=((1, 1, 3, 1, 0, 2, 1, 0),),
                        )

    parser.add_argument('--dropout',
                        type=bool,
                        default=False,
                        )

    parser.add_argument('--batch-norm',
                        type=bool,
                        default=False,
                        )

    parser.add_argument('--learning-rate',
                        type=float,
                        default=0.0001,
                        )

    parser.add_argument('--activation',
                        type=str,
                        default='ReLU',
                        )

    parser.add_argument('--activation-params',
                        type=dict,
                        default={},
                        )

    parser.add_argument('--verbose',
                        type=bool,
                        default=False,
                        )

    args = parser.parse_args()

    if args.model_type not in ['fc', 'cnn', 'res']:
        print('Model type not supported, only one of'
              ' "fc", "cnn", "res" allowed')
        sys.exit(1)

    if not all([len(x) == 8 for x in args.cnn_options]):
        print('Invalid CNN options passed in: was missing argument')
        sys.exit(1)

    if args.activation not in torch.nn.__dict__.keys():
        print('Invalid activation passed in: does not exist')
        sys.exit(1)

    if args.n_epochs < 1:
        print('Invalid number of epochs passed in: must be greater than 1')
        sys.exit(1)

    if args.batch_size < 1:
        print('Invalid batch size passed in: must be greater than 1')
        sys.exit(1)

    if args.learning_rate < 0:
        print('Invalid learning rate passed in: must be positive')
        sys.exit(1)

    train_dataset = CSGODataset(transform=transform_data,
                                dataset_split='train',
                                verbose=args.verbose,
                                )

    val_dataset = CSGODataset(transform=transform_data,
                              dataset_split='val',
                              verbose=args.verbose,
                              )

    test_dataset = CSGODataset(transform=transform_data,
                               dataset_split='test',
                               verbose=args.verbose,
                               )

    if len(sys.argv) < 2:
        test_train_functions(train_dataset, val_dataset, test_dataset)
        sys.exit()

    # implicit else
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=0,
                                               )

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=0,
                                             )

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=args.batch_size,
                                              shuffle=False,
                                              num_workers=0,
                                              )

    models = {'fc': FCNN, 'cnn': CNN, 'res': ResNet}

    model = models[args.model_type](hidden_sizes=args.hidden_sizes,
                                    activation=args.activation,
                                    activation_params=args.activation_params,
                                    dropout=args.dropout,
                                    batch_norm=args.batch_norm,
                                    cnn_options=args.cnn_options,
                                    )

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_fn = torch.nn.BCELoss()

    for i in range(args.n_epochs):
        print('\n' + '=' * 30)
        print(f'Training epoch {i + 1}')
        train(model=model,
              loader=train_loader,
              optimizer=optimizer,
              loss_fn=loss_fn,
              device=device,
              verbose=args.verbose,
              )

        test(model=model,
             loader=val_loader,
             device=device,
             )

    print('\n\n\n' + '+' * 30)
    print(f'Test set results for: {args}\n\n')

    test(model=model,
         loader=test_loader,
         device=device,
         )

    random_number = random.random()

    torch.save(model.state_dict(), f'model-{random_number:.5f}.pt')

    print(f'Saved to model-{random_number:.5f}.pt')
