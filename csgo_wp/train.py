#! /usr/bin/python3

import torch
from model import CNNModel, FCModel  # noqa


def train(model, data, targets, optimizer, loss_fn, device):
    model.train()
    model.to(device)

    total_loss = 0

    for i, subset in data.groupby(['MatchId', 'MapName', 'RoundNum']):
        print(f'Subset {i}')
        t = transform_data(data, i[1]).to(device)

        target = (targets[(targets['MatchId'] == i[0]) &
                          (targets['MapName'] == i[1]) &
                          (targets['RoundNum'] == i[2])
                          ]['RoundWinnerSide'] == 'CT')

        target = torch.full(t.shape[:1], target.astype(int).values[0]).long()
        target = target.to(device)

        optimizer.zero_grad()

        output = model(t)

        loss = loss_fn(output, target)

        total_loss += loss.item()

        loss.backward()

        optimizer.step()

    print(f'Total loss: {total_loss}')


if __name__ == '__main__':
    import pandas as pd
    from data_transform import transform_data

    rowlim = 15000
    data = pd.read_csv('/home/gpt/Desktop/example_frames.csv', nrows=rowlim)

    targets = pd.read_csv('/home/gpt/Desktop/example_rounds.csv', nrows=rowlim)
    targets = targets[['MatchId', 'MapName', 'RoundNum', 'RoundWinnerSide']]

    device = 'cpu'

    mod = CNNModel().to(device)
    # mod = FCModel().to(device)

    optimizer = torch.optim.Adam(mod.parameters())
    loss_fn = torch.nn.CrossEntropyLoss()

    for i in range(5):
        print(f'Training epoch {i + 1}')
        train(model=mod,
              data=data,
              targets=targets,
              optimizer=optimizer,
              loss_fn=loss_fn,
              device=device,
              )
