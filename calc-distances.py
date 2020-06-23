#! /usr/bin/python3

from csgo.analytics.distance import area_distance
import time
import queue


def calc_area_distance(params_q, distances_q):
    while True:
        try:
            params = params_q.get(timeout=5)
        except queue.Empty:
            print('Params queue empty')
            break

        distance = area_distance(params['map'], params['a1'], params['a2'])

        distances_q.put((params['map'], params['a1'], params['a2'], distance))

    return


def write_distances(q, count):
    from os import path

    written = 0
    start_time = time.time()

    # one folder up, into the 'data' folder
    current_dir = path.dirname(path.abspath(__file__))
    file_loc = path.join(current_dir, '..', 'data', 'distance_infos.csv')

    colnames = ['map', 'areaId_1', 'areaId_2', 'graph_distance']

    with open(file_loc, 'w') as f:
        f.write(','.join(colnames) + '\n')

        time.sleep(30)
        while written < (8 * 8) + 2:
            try:
                infos = q.get(timeout=5)
            except queue.Empty:
                print('    Distances queue empty!    ')
                time.sleep(30)
                continue

            f.write(','.join(infos) + '\n')
            written += 1
            if written % 1000 == 0:
                print(f'\rCalculated {written}/{count.value} in '
                      f'{time.time() - start_time}')

        print('exiting write_distances')
    return


if __name__ == '__main__':
    import multiprocessing

    params = multiprocessing.Queue()
    distances = multiprocessing.Queue()

    count = multiprocessing.Value('i', 0)
    infos_q = multiprocessing.Queue()

    for i in range(1, 9):
        for j in range(1, 9):
            params.put({'map': 'de_mirage',
                        'a1': i,
                        'a2': j,
                        })

    procs = {}

    for p in range(0, 10):
        procs[p] = multiprocessing.Process(target=calc_area_distance,
                                           args=(params,
                                                 distances,
                                                 ))
        procs[p].start()

    # initialize writer
    p2 = multiprocessing.Process(target=write_distances,
                                 args=(distances, count))
    p2.start()

    for p in range(0, 10):
        procs[p].join()

    p2.join()
