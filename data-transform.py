#! /usr/bin/python3

import torch
from csgo.analytics.distance import point_distance, area_distance
from functools import partial


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
        df[f'pos{c}'] = df[f'pos{c}'].astype(float)
        df['areaId'] = df['areaId'].astype(int)

    df['hp'] = df['hp'].astype(int)

    df = df[['side',
             'hp',
             'name',
             'posX',
             'posY',
             'posZ',
             'tick',
             'areaId',
             ]]
    df['pos'] = df[['posX', 'posY', 'posZ']].values.tolist()

    merged = df.merge(df, on='tick')
    merged['pos_diff'] = merged[['pos_x', 'pos_y']].values.tolist()
    merged['area_diff'] = merged[['areaId_x', 'areaId_y']].values.tolist()

    area_dist = partial(area_dist_all, game_map=game_map)

    distance_matrix = merged.pivot_table(index=['tick', 'name_x'],
                                         columns='name_y',
                                         values='area_diff',
                                         aggfunc=area_dist,
                                         )

    t = torch.Tensor(distance_matrix.values).view(-1, 10, 10)

    additional_data = (df.groupby(['tick', 'name'])
                         .agg({'hp': is_alive,  # how about returning hp?
                               'side': is_ct,
                               })
                         .astype(int)
                       )

    t_2 = torch.Tensor(additional_data.values).view(-1, 10, 2)

    return t, t_2
