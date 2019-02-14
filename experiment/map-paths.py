#!/usr/bin/python

import sys
import retro
import numpy as np
from os import listdir
from os.path import isfile, join, isdir, dirname, realpath
from PIL import Image


# find level maps here: http://info.sonicretro.org/Sonic_the_Hedgehog_(16-bit)_level_maps
mp = Image.open(dirname(realpath(__file__))+"/Lz1.png")
mp.load()
level_map =np.array(mp.convert(mode='RGB'), dtype="uint8" )
print('map shape', level_map.shape)
hf = 10 # highlight factor

def render(file):
    movie = retro.Movie(file)
    movie.step()
    env = retro.make(game=movie.get_game(), state=None, use_restricted_actions=retro.Actions.ALL, players=movie.players)
    env.initial_state = movie.get_state()
    env.reset()
    while movie.step():
        keys = []
        for p in range(movie.players):
            for i in range(env.num_buttons):
                keys.append(movie.get_key(i, p))
        _obs, _rew, _done, _info = env.step(keys)
        y = _info['y']
        x = _info['x'] 
        highlight = [[[min(x[0][0]+hf,255), min(x[0][1]+hf,255), min(x[0][2]+hf,255)],
        [min(x[1][0]+hf,255), min(x[1][1]+hf,255), min(x[1][2]+hf,255)],
        [min(x[2][0]+hf,255), min(x[2][1]+hf,255), min(x[2][2]+hf,255)],
        [min(x[3][0]+hf,255), min(x[3][1]+hf,255), min(x[3][2]+hf,255)],
        [min(x[4][0]+hf,255), min(x[4][1]+hf,255), min(x[4][2]+hf,255)],
        [min(x[5][0]+hf,255), min(x[5][1]+hf,255), min(x[5][2]+hf,255)],
        [min(x[6][0]+hf,255), min(x[6][1]+hf,255), min(x[6][2]+hf,255)],
        [min(x[7][0]+hf,255), min(x[7][1]+hf,255), min(x[7][2]+hf,255)],
        ] for x in level_map[y:(y+8), x:(x+8)]]

        level_map[y:(y+8), x:(x+8)] = highlight

    env.close()

if isdir(sys.argv[1]):
    onlyfiles = [f for f in listdir(sys.argv[1]) if isfile(join(sys.argv[1], f))]
    onlyfiles.sort()
    c = 0
    for file in onlyfiles:
        if ".bk2" in file :
            print('playing', file)
            try:
                render(sys.argv[1]+file)
            except:
                c+= 1
                continue
        if c % 1==0:
            lm = Image.fromarray(level_map)
            lm.save('levelmap.jpeg')
        c+=1
    lm.show()
else:
    print('playing', sys.argv[1])
    render(sys.argv[1])
    lm = Image.fromarray(level_map)
    # lm.save('levelmap.jpeg')
lm.show()