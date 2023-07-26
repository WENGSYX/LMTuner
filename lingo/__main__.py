from lingo import let_lingo,get_cmd,launch_cmd
import json
import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--ARGS', type=str, default='')
    args = parser.parse_args()

    let_lingo(args.ARGS)
