#! /usr/bin/env python3

import torch
from model import FCNN, CNN, ResNet  # noqa
from sklearn.metrics import log_loss, roc_auc_score, accuracy_score


def train(model, loader, optimizer, loss_fn, device):
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

        print('\n' + '=' * 30)
        print('Results')
        print(f'Accuracy: {accuracy_score(y_true, y_pred > 0.5):.4f}')
        print(f'AUC: {roc_auc_score(y_true, y_pred):.4f}')
        print(f'Log loss: {log_loss(y_true, y_pred):.4f}')


if __name__ == '__main__':
    from data_transform import CSGODataset, transform_data

    train_dataset = CSGODataset(transform=transform_data,
                                dataset_split='train')

    val_dataset = CSGODataset(transform=transform_data,
                              dataset_split='val')

    test_dataset = CSGODataset(transform=transform_data,
                               dataset_split='test')

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
        print('\n' + '-' * 30)
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

    print('\n\n\n' + '=' * 30)
    print('Test set results')
    test(model=model,
         loader=test_loader,
         device=device,
         )
