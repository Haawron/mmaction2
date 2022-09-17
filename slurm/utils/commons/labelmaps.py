from cProfile import label


labelmaps = {
    'ek100': [
        'take',
        'put',
        'wash',
        'open',
        'close',
        'unk',
    ],
    'ucf': [
        'climb',            # 0
        'fencing',          # 1
        'golf',             # 2
        'kick-ball',        # 3
        'pullup',           # 4

        'punch',            # 5
        'pushup',           # 6
        'ride-bike',        # 7
        'ride-horse',       # 8
        'shoot-ball',       # 9

        'shoot-bow',        # 10
        'walk',             # 11
        'unk',
    ],
    'hmdb': [
        'climb',            # 0
        'fencing',          # 1
        'golf',             # 2
        'kick-ball',        # 3
        'pullup',           # 4

        'punch',            # 5
        'pushup',           # 6
        'ride-bike',        # 7
        'ride-horse',       # 8
        'shoot-ball',       # 9

        'shoot-bow',        # 10
        'walk',             # 11
        'unk',
    ],
}

labelmaps['hmdb'] = labelmaps['ucf']
