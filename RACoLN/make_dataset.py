import os
import sys
from src.util import create_json
import argparse
import logging

logging.basicConfig(
format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
datefmt="%Y-%m-%d %H:%M:%S",
level=os.environ.get("LOGLEVEL", "INFO").upper(),
stream=sys.stdout,
)
logger = logging.getLogger("make_dataset.py")


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='jigsaw')
    parser.add_argument('--phase', type=str, nargs='+')
    args = parser.parse_args()

    for phase in args.phase:
        print(f'making dataset: {args.data} - {phase}')
        create_json(args.data, phase)