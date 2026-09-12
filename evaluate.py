"""Use the identical task-head PPL and metrics code as sample.py."""
import sys
from sample import main

if __name__ == '__main__':
    if '--input' not in sys.argv and '--help' not in sys.argv and '-h' not in sys.argv:
        raise SystemExit('Usage: python evaluate.py --task amp --input sequences.fasta')
    main()
