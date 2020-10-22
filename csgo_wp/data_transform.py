#! /usr/bin/python3

import torch
from csgo.analytics.distance import point_distance, area_distance
from functools import partial
import os
from pathlib import Path
import pandas as pd


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
    df['Hp'] = df['Hp'].astype(int)

    # this better not be a fractional
    players_count = df['SteamId'].nunique()

    df = df[['Side',
             'Hp',
             'SteamId',
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

    distance_matrix = merged.pivot_table(index=['Tick', 'SteamId_x'],
                                         columns='SteamId_y',
                                         values='area_diff',
                                         aggfunc=area_dist,
                                         )

    t = torch.Tensor(distance_matrix.values).view(-1,
                                                  players_count,
                                                  players_count)

    additional_data = (df.groupby(['Tick', 'SteamId'])
                         .agg({'Hp': is_alive,  # how about returning hp?
                               'Side': is_ct,
                               })
                         .astype(int)
                       )

    t_2 = torch.Tensor(additional_data.values).view(-1, players_count, 2)

    # return torch.cat((t, t_2), dim=2).unsqueeze(1)
    n_samples = t.shape[0]
    result = torch.cat([t.reshape(n_samples, players_count ** 2),
                        t_2.reshape(n_samples, players_count * 2)],
                       dim=1).view(n_samples, players_count + 2, players_count)

    return result


class CSGODataset(torch.utils.data.Dataset):

    def __init__(self,
                 folder='G:/datasets/csgo/',
                 transform=None,
                 dataset_split='train'):
        self.folder = folder + 'match-map-unique/'
        self.split = dataset_split
        self.split_folder = self.folder + dataset_split

        if transform is None:
            raise ValueError('Transform required')

        if not os.path.exists(self.folder):
            print('Train/val/test splits not found')
            self.file_loc = folder + 'example_frames.csv'
            # parse csv and separate into folder

            os.makedirs(self.folder + 'train')
            os.makedirs(self.folder + 'val')
            os.makedirs(self.folder + 'test')

            print('Loading entire dataframe into memory...')
            df = pd.read_csv(self.file_loc)

            print('Getting match/map/round combinations...')
            # list of lists
            match_map_combos = (df[['MatchId',
                                    'MapName',
                                    'RoundNum',
                                    ]].drop_duplicates()
                                      .values.tolist())

            print('Writing combinations to train/val/test folders...')
            for combo in match_map_combos:
                value = torch.rand(1).item()

                if value > 0.8:
                    split = 'test'
                elif value < 0.6:
                    split = 'train'
                else:
                    split = 'val'

                subset = df[(df['MatchId'] == combo[0])
                            & (df['MapName'] == combo[1])
                            & (df['RoundNum'] == combo[2])].copy()
                tick_count = subset['Tick'].nunique()
                subset.to_csv(f'{self.folder}{split}/match-{combo[0]}-'
                              f'{combo[1]}-{combo[2]}-{tick_count}.csv')
            print('Data written to disk')

        self.transform = transform

        self.folder = Path(self.split_folder)
        self.matchups = list(self.folder.glob('*.csv'))

        self.matchup_idx_by_sample_idx = {}
        self.n_samples = 0

        print('Calculating sample index offsets...')

        for idx, matchup in enumerate(self.matchups):
            tick_count = int(matchup.stem.split('-')[-1])  # last component

            for i in range(self.n_samples, self.n_samples + tick_count):
                self.matchup_idx_by_sample_idx[i] = (idx, self.n_samples)

            self.n_samples += tick_count
        print('Done!')

        self.targets = pd.read_csv(folder + 'example_rounds.csv',
                                   usecols=['MatchId',
                                            'MapName',
                                            'RoundNum',
                                            'RoundWinnerSide',
                                            ])

    def __len__(self):
        return self.n_samples

    def __getitem__(self, sample_idx):
        idx, n_samples_prior = self.matchup_idx_by_sample_idx[sample_idx]

        idx_in_match = sample_idx - n_samples_prior

        df = pd.read_csv(self.matchups[idx])
        match_id = df['MatchId'].values[0]
        map_name = df['MapName'].values[0]
        round_num = df['RoundNum'].values[0]

        transformed_df = self.transform(df, map_name)
        transformed_df = transformed_df[idx_in_match]

        data = torch.Tensor(transformed_df)

        round_result = self.targets[(self.targets['MatchId'] == match_id)
                                    & (self.targets['MapName'] == map_name)
                                    & (self.targets['RoundNum'] == round_num)]

        target = (round_result['RoundWinnerSide'] == 'CT').astype(int)
        target = torch.Tensor(target.values)

        return data, target.view(1)


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
