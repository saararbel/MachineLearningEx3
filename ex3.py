import sys
import numpy as np

# def train(train_file_path, gain_measure):
#     return id3_algorithm(examples, attributes, attributes_values, gain_measure)

def parse_train_file(train_file_path):
    text = np.loadtxt(train_file_path, dtype=str)
    attributes = text[0].tolist()
    examples = text[1:]
    attributes_values = {}
    for attribute in attributes:
        attributes_values[attribute] = set()
    for example in examples:
        for i, value in enumerate(example):
            attribute = attributes[i]
            attributes_values[attribute].add(value)
    return attributes_values


if __name__ == '__main__':
    train_file_path = sys.argv[1]
    validation_file_path = sys.argv[2]
    measure_gain = sys.argv[3]
    a = parse_train_file(train_file_path)
    print measure_gain