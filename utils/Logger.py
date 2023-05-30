import os
import sys
import shutil
import builtins


# log ------------------------------------
def remove_comments(lines, token='#'):
    """ Generator. Strips comments and whitespace from input lines.
    """

    l = []
    for line in lines:
        s = line.split(token, 1)[0].strip()
        if s != '':
            l.append(s)
    return l


def open(file, mode=None, encoding=None):
    if mode == None: mode = 'r'

    if '/' in file:
        if 'w' or 'a' in mode:
            dir = os.path.dirname(file)
            if not os.path.isdir(dir):  os.makedirs(dir)

    f = builtins.open(file, mode=mode, encoding=encoding)
    return f


def remove(file):
    if os.path.exists(file): os.r