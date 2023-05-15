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

    'k400': [
        'jump',             # 0
        'run',              # 1
        'throw',            # 2
        'kick',             # 3
        'bend',             # 4
        'dance',            # 5
        'clean_something',  # 6
        'squat',            # 7
        'punch',            # 8
        'crawl',            # 9
        'clap',             # 10
        'pick_up',          # 11
    ],

    'babel': [
        'jump',             # 0
        'run',              # 1
        'throw',            # 2
        'kick',             # 3
        'bend',             # 4
        'dance',            # 5
        'clean_something',  # 6
        'squat',            # 7
        'punch',            # 8
        'crawl',            # 9
        'clap',             # 10
        'pick_up',          # 11

        'walk',             # 12
        'turn',             # 13
        'step',             # 14
        'sit down',         # 15
        'stand up',         # 16
        'wave',             # 17
        'place',            # 18
        'catch',            # 19
    ],
}

labelmaps['hmdb'] = labelmaps['ucf']
labelmaps['kinetics400'] = labelmaps['k400']
