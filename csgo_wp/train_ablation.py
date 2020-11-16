#! /usr/bin/env python3

import torch
from model import LR_CNN
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

        y_pred = torch.cat(outputs, dim=0).cpu().numpy().astype(float)
        y_true = torch.cat(targets, dim=0).cpu().numpy().astype(float)

        print('\n' + '-' * 30)
        print('Results')
        print(f'Accuracy: {accuracy_score(y_true, y_pred > 0.5):.4f}')
        print(f'AUC: {roc_auc_score(y_true, y_pred):.4f}')
        print(f'Log loss: {log_loss(y_true, y_pred):.4f}')

    return roc_auc_score(y_true, y_pred)


if __name__ == '__main__':
    from data_transform import CSGODataset, transform_multichannel
    from torch.utils.data import ConcatDataset
    import sys
    import argparse
    import warnings
    warnings.filterwarnings('ignore')

    parser = argparse.ArgumentParser()

    parser.add_argument('--ablation',
                        type=str,
                        default=None,
                        )

    args = parser.parse_args()

    if args.ablation not in [None, 'distance', 'player_count']:
        print('Ablation type not supported, only one of'
              ' "distance", "player_count" allowed')
        sys.exit(1)

    train_dataset = CSGODataset(transform=transform_multichannel,
                                dataset_split='train',
                                verbose=False,
                                )

    val_dataset = CSGODataset(transform=transform_multichannel,
                              dataset_split='val',
                              verbose=False,
                              )

    test_dataset = CSGODataset(transform=transform_multichannel,
                               dataset_split='test',
                               verbose=False,
                               )

    train_val_dataset = ConcatDataset([train_dataset, val_dataset])

    # implicit else
    train_loader = torch.utils.data.DataLoader(train_val_dataset,
                                               batch_size=64,
                                               shuffle=True,
                                               num_workers=0,
                                               )

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=64,
                                              shuffle=False,
                                              num_workers=0,
                                              )

    model = LR_CNN(input_size=(6, 5, 5),
                   hidden_sizes=[200, 100, 50],
                   activation='LeakyReLU',
                   activation_params={},
                   dropout=False,
                   batch_norm=False,
                   cnn_options=((4, 6, 1, 1, 0, 1, 1, 0),
                                (6, 6, 1, 1, 0, 1, 1, 0),
                                (6, 6, 5, 1, 0, 1, 1, 0),),
                   ablation=args.ablation,
                   )

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)
    loss_fn = torch.nn.BCELoss()

    aucs = {}
    print(f'Training for {args.ablation} ablation')
    for i in range(22):
        print('\n' + '=' * 30)
        print(f'Training epoch {i + 1}')
        train(model=model,
              loader=train_loader,
              optimizer=optimizer,
              loss_fn=loss_fn,
              device=device,
              verbose=False,
              )

    print('\n\n\n' + '+' * 30)
    print(f'Test set results\n\n')

    test(model=model,
         loader=test_loader,
         device=device,
         )

    torch.save(model.state_dict(), f'model-{args.ablation}.pt')

    print(f'Saved to model-{args.ablation}.pt')
