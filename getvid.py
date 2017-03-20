#!/usr/bin/env python

import sys
import subprocess
import csv
import os

from concurrent.futures import ProcessPoolExecutor


def get_vid(video, offset, duration):
    try:
        subprocess.run(["cvlc",
                        "--start-time", "%.2f" % offset,
                        "--stop-time", "%.2f" % (offset + duration),
                        "--video-filter", "scene",
                        "--scene-format", "jpeg",
                        "--scene-path", "./",
                        "--scene-ratio", "1",
                        "-V", "dummy",
                        "https://www.youtube.com/watch?v=%s" % video],
                       timeout=duration + 5)
    except Exception as e:
        pass

def do_thing(t):
    line, rec = t
    line += 1
    print(line)
    vid = rec[0]
    offset = float(rec[5]) / 1000.0
    print("vid: %s, offset: %.2f" % (vid, offset))
    dir = str(line)
    os.mkdir(dir)
    os.chdir(dir)

    # half second back, don't miss start
    offset -= 0.5
    if offset < 0.0:
        offset = 0.0

    get_vid(vid, offset, 4)

    os.chdir("..")

def main():
    with open(sys.argv[1]) as f:
            r = csv.reader(f)
            with ProcessPoolExecutor(16) as exec:
                fs = exec.map(do_thing, enumerate(r))
                

if __name__ == '__main__':
    main()
