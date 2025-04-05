import dataset_formatter.formatter as formatter
import sys
import os

if __name__ == '__main__':
    try:
        path = sys.argv[4]
        dataset_name = sys.argv[1]
        num_sources = int(sys.argv[2])
        alpha = int(sys.argv[3])
    except:
        raise SyntaxError('Usage: python agent_dataset.py <identifier> <num_sources> <alpha> <path>')

    formatter.prepare_data(path, dataset_name, num_sources, alpha)
    # formatter.prepare_data('Rowan/hellaswag', args[1], args[2])
