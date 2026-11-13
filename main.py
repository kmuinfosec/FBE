import sys
from utils.config import load
from utils import preprocessing, experiment, summary

COMMANDS = {"preprocess": preprocessing.run, "run": experiment.run, "summary": summary.run}


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        sys.exit("usage: python main.py preprocess <dataset> | run <lgbm|cnn> <dataset> <input> <seed> | summary")
    COMMANDS[sys.argv[1]](load(), *sys.argv[2:])


if __name__ == "__main__":
    main()
