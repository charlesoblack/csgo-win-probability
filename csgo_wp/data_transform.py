#! /usr/bin/env python3

import torch
from csgo.analytics.distance import point_distance, area_distance
from functools import partial
import os
import pandas as pd
from collections import defaultdict
import pickle
import numpy as np


def euclidean_distance(x, game_map):
    return point_distance(x.values[0][0],
                          x.values[0][1],
                          map=game_map,
                          type='euclidean',
                          )


def area_dist_all(x, game_map):
    return area_distance(area_one=x.values[0][0],
                         area_two=x.values[0][1],
                         map=game_map,
                         )


def is_alive(x):
    return x > 0


def is_ct(x):
    return x == 'CT'


def transform_data(df, game_map):
    df.drop_duplicates(inplace=True)

    for c in ['X', 'Y', 'Z']:
        df[c] = df[c].astype(float)

    df['AreaId'] = df['AreaId'].astype(int)
    df['IsAlive'] = df['IsAlive'].astype(bool)

    df = df[['Side',
             'IsAlive',
             'PlayerSteamId',
             'X',
             'Y',
             'Z',
             'Tick',
             'AreaId',
             ]]
    # TODO: fix SettingWithCopy warning
    df['pos'] = df[['X', 'Y', 'Z']].values.tolist()

    merged = df.merge(df, on='Tick')
    merged['pos_diff'] = merged[['pos_x', 'pos_y']].values.tolist()
    merged['area_diff'] = merged[['AreaId_x', 'AreaId_y']].values.tolist()

    area_dist = partial(area_dist_all, game_map=game_map)

    distance_matrix = merged.pivot_table(index=['Tick', 'PlayerSteamId_x'],
                                         columns='PlayerSteamId_y',
                                         values='area_diff',
                                         aggfunc=area_dist,
                                         )

    t = torch.Tensor(distance_matrix.values).view(-1, 10, 10)

    additional_data = (df.groupby(['Tick', 'PlayerSteamId'])
                         .agg({'IsAlive': 'any',  # how about returning hp?
                               'Side': is_ct,
                               })
                         .astype(int)
                       )

    t_2 = torch.Tensor(additional_data.values).view(-1, 10, 2)

    # return torch.cat((t, t_2), dim=2).unsqueeze(1)
    n_samples = t.shape[0]
    result = torch.cat([t.reshape(n_samples, 100),
                        t_2.reshape(n_samples, 20)],
                       dim=1).view(n_samples, 12, 10)

    return result


def transform_multichannel(df, game_map):
    df.drop_duplicates(inplace=True)

    for c in ['X', 'Y', 'Z']:
        df[c] = df[c].astype(float)

    df['AreaId'] = df['AreaId'].astype(int)
    df['IsAlive'] = df['IsAlive'].astype(bool)

    df = df[['Side',
             'IsAlive',
             'PlayerSteamId',
             'X',
             'Y',
             'Z',
             'Tick',
             'AreaId',
             ]]
    # TODO: fix SettingWithCopy warning
    df['pos'] = df[['X', 'Y', 'Z']].values.tolist()

    merged = df.merge(df, on='Tick')
    merged['pos_diff'] = merged[['pos_x', 'pos_y']].values.tolist()
    merged['area_diff'] = merged[['AreaId_x', 'AreaId_y']].values.tolist()

    area_dist = partial(area_dist_all, game_map=game_map)

    distance_matrix = merged.pivot_table(index=['Tick', 'PlayerSteamId_x'],
                                         columns='PlayerSteamId_y',
                                         values='area_diff',
                                         aggfunc=area_dist,
                                         )

    t_players = np.sort(df[df['Side'] == 'T']['PlayerSteamId']
                        .unique()
                        ).tolist()
    ct_players = np.sort(df[df['Side'] == 'CT']['PlayerSteamId']
                         .unique()
                         ).tolist()

    distance_matrix.reset_index(level=1, drop=False, inplace=True)

    t_rows = distance_matrix['PlayerSteamId_x'].isin(t_players)
    ct_rows = distance_matrix['PlayerSteamId_x'].isin(ct_players)

    # all 4 combinations
    channel_1 = torch.Tensor(distance_matrix[t_rows][t_players].values)
    channel_2 = torch.Tensor(distance_matrix[ct_rows][ct_players].values)
    channel_3 = torch.Tensor(distance_matrix[t_rows][ct_players].values)
    channel_4 = torch.Tensor(distance_matrix[ct_rows][t_players].values)

    # (batch_size, 4, 5, 5)
    t = torch.stack([channel_1.view(-1, 5, 5),
                     channel_2.view(-1, 5, 5),
                     channel_3.view(-1, 5, 5),
                     channel_4.view(-1, 5, 5),
                     ],
                    dim=1)

    additional_data_t = (df[df['Side'] == 'T']
                         .groupby(['Tick', 'PlayerSteamId'], as_index=False)
                         .agg({'IsAlive': 'any',  # how about returning hp?
                               })
                         .astype(int)
                         .sort_values(by=['Tick', 'PlayerSteamId'])
                         )

    additional_data_ct = (df[df['Side'] == 'CT']
                          .groupby(['Tick', 'PlayerSteamId'], as_index=False)
                          .agg({'IsAlive': 'any',
                                })
                          .astype(int)
                          .sort_values(by=['Tick', 'PlayerSteamId'])
                          )

    # (batch_size, 1, 5, 5)
    t_2 = (torch.stack([torch.diag(torch.Tensor(x['IsAlive'].values))
                        for idx, x in additional_data_t.groupby(['Tick'])])
                .view(-1, 1, 5, 5))
    t_3 = (torch.stack([torch.diag(torch.Tensor(x['IsAlive'].values))
                        for idx, x in additional_data_ct.groupby(['Tick'])])
                .view(-1, 1, 5, 5))

    result = torch.cat([t, t_2, t_3],
                       dim=1).view(-1, 6, 5, 5)

    return result


class CSGODataset(torch.utils.data.Dataset):

    def __init__(self,
                 folder='G:/datasets/csgo/',
                 transform=None,
                 dataset_split='train',
                 verbose=False,
                 rng_seed=13):
        self.rng_seed = rng_seed
        torch.manual_seed(rng_seed)
        np.random.seed(rng_seed)

        self.split = dataset_split

        bad_round_count = 0

        if transform is None:
            raise ValueError('Transform required')

        if not os.path.exists(folder + 'test'):
            print('Train/val/test splits not found')

            self.file_loc = folder + 'csgo_playerframes_dust2.csv'

            os.makedirs(folder + 'train')
            os.makedirs(folder + 'val')
            os.makedirs(folder + 'test')

            print('Loading entire dataframe into memory...')

            frames_columns = ['MatchId',
                              'MapName',
                              'RoundNum',
                              'Tick',
                              'Second',
                              'PlayerId',
                              'PlayerSteamId',
                              'TeamId',
                              'Side',
                              'X',
                              'Y',
                              'Z',
                              'ViewX',
                              'ViewY',
                              'AreaId',
                              'Hp',
                              'Armor',
                              'IsAlive',
                              'IsFlashed',
                              'IsAirborne',
                              'IsDucking',
                              'IsScoped',
                              'IsWalking',
                              'EqValue',
                              'HasHelmet',
                              'HasDefuse',
                              'DistToBombsiteA',
                              'DistToBombsiteB',
                              'Created',
                              'Updated']

            # only load required columns in
            df = pd.read_csv(self.file_loc,
                             names=frames_columns,
                             usecols=['MatchId',
                                      'MapName',
                                      'RoundNum',
                                      'Tick',
                                      'PlayerSteamId',
                                      'X',
                                      'Y',
                                      'Z',
                                      'AreaId',
                                      'IsAlive',
                                      'Side',
                                      ])

            print('Getting match/map combinations...')
            # list of lists
            match_map_combos = (df[['MatchId',
                                    'MapName',
                                    ]].drop_duplicates()
                                      .values.tolist())

            splits = defaultdict(list)

            print('Dropping bogus rounds...')
            for combo in match_map_combos:
                value = torch.rand(1).item()

                if value > 0.8:
                    split = 'test'
                elif value < 0.6:
                    split = 'train'
                else:
                    split = 'val'

                subset = df[(df['MatchId'] == combo[0])
                            & (df['MapName'] == combo[1])].copy()

                for round_num in subset['RoundNum'].unique():
                    game_round = subset[subset['RoundNum'] == round_num]
                    player_count = game_round['PlayerSteamId'].nunique()
                    tick_count = game_round['Tick'].nunique()
                    tick_x_players = player_count * tick_count

                    if (player_count < 10
                       or game_round.shape[0] != tick_x_players):
                        # if we have less than 10 players, ignore this df
                        # hopefully this doesn't affect the train/test split
                        # ratio too much
                        bad_round_count += 1
                        # drop the rows with the bogus round
                        continue

                    splits[split].append(game_round)

            print(f'Found {bad_round_count} rounds with fewer than 10 players')

            for k, v in splits.items():
                with open(folder + k + f'/{k}.pckl', 'wb') as f:
                    pickle.dump(v, f)

            self.raw_data = splits[self.split]
            del splits
        else:
            with open(f'{folder}{self.split}/{self.split}.pckl',
                      'rb') as f:
                self.raw_data = pickle.load(f)

        self.transform = transform
        self.data = []
        self.targets = []

        self.rounds = pd.read_csv(folder + 'csgo_rounds_dust2.csv',
                                  usecols=['MatchId',
                                           'MapName',
                                           'RoundNum',
                                           'WinningSide',
                                           ])

        print('Transforming raw data...')

        len_data = len(self.raw_data)

        for idx, game_round in enumerate(self.raw_data):
            match_id = game_round['MatchId'].values[0]
            map_name = game_round['MapName'].values[0]
            round_num = game_round['RoundNum'].values[0]

            if verbose:
                print(f'\rTransforming {idx +1}/{len_data}: {match_id}, '
                      f'{map_name}, {round_num}  ', end='')
            transformed = self.transform(game_round, 'de_dust2')
            self.data.extend(transformed)

            target = self.rounds[(self.rounds['MatchId'] == match_id)
                                 & (self.rounds['MapName'] == map_name)
                                 & (self.rounds['RoundNum'] == round_num)]
            target = 1 if target['WinningSide'].iloc[0] == 'CT' else 0
            self.targets.extend([target
                                 for _ in range(transformed.shape[0])])

        self.data = torch.stack(self.data)
        self.targets = torch.Tensor(self.targets)

        print('\nDone!')

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


if __name__ == '__main__':
    dataset = CSGODataset(transform=transform_data)

    loader = torch.utils.data.DataLoader(dataset,
                                         batch_size=1,
                                         shuffle=True,
                                         num_workers=0,
                                         )

    # check if dataset acts as expected
    for index, (data, target) in enumerate(loader):
        print(index, data.shape, target.shape)
        break
