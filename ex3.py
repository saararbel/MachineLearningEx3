import sys
from math import log
import numpy as np
from collections import Counter


# def train(train_file_path, gain_measure):
#     return id3_algorithm(examples, attributes, attributes_values, gain_measure)



def write_final_tree(tree, number_tabs,output_file):
    for child in tree.get_node_sub_trees():
        for i in xrange(number_tabs):
            output_file.write(" ")
        if child.is_have_sub_trees() == False:
            print str(tree.node_attribute) + " = " + str(child.get_value()) + " : " + str(child.get_node_attribute())
            output_file.write(str(tree.node_attribute) + " = " + str(child.get_value()) + " : " + str(child.get_node_attribute()))
        else :
            print str(tree.node_attribute) + " = " + child.get_value()
            output_file.write(str(tree.node_attribute) + " = " + child.get_value())
        output_file.write("\n")

        write_final_tree(child, number_tabs+1, output_file)


def id3_algorithm(examples, attributes, attributes_values, gain_measure, default=None, value="ROOT"):
    # there is no exmaples
    if len(examples) == 0:
        return Node(default, value)

    # all the exmaples are from the same classification
    same_classification = True
    first_classification = examples[0][-1]
    for e in examples:
        if e[-1] != first_classification:
            same_classification = False
            break
    if same_classification:
        return Node(first_classification, value)

    elif len(attributes[:-1]) == 0:
        return Node(majority_value(examples), value)
    else:
        best_attribute = choose_best_attribute(attributes, examples, gain_measure)
        tree = Node(best_attribute, value)
        examples_by_value = split_examples_by_value(examples, attributes.index(best_attribute),
                                                    attributes_values[best_attribute])
        for value, example_i in examples_by_value.iteritems():
            subtree = id3_algorithm(example_i, [a for a in attributes if a != best_attribute],
                                    attributes_values,
                                    gain_measure,
                                    majority_value(examples), value)
            tree.insert_sub_tree(subtree)
        return tree


def majority_value(examples):
    return Counter(examples[:, -1]).most_common(1)[0][0]


def split_examples_by_value(examples, attribute_index, attribute_values):
    examples_by_value = {}
    for value in attribute_values:
        examples_by_value[value] = []
    for example in examples:
        value = example[attribute_index]
        examples_by_value[value].append(list_without(example, attribute_index))
    for value in examples_by_value:
        examples_by_value[value] = np.array(examples_by_value[value])
    return examples_by_value


def list_without(a_list, i):
    return np.delete(a_list, i)


def parse_train_file(train_file_path):
    text = np.loadtxt(train_file_path, dtype=str)
    attributes_in_file = text[0].tolist()
    attributes_to_values = {}
    for attribute in attributes_in_file:
        attributes_to_values[attribute] = set()

    examples = text[1:]
    for example in examples:
        for idx, value in enumerate(example):
            attribute = attributes_in_file[idx]
            attributes_to_values[attribute].add(value)

    return attributes_to_values, attributes_in_file, examples


def choose_best_attribute(attributes, examples, gain_measure):
    probabilietes = [float(count) / len(examples) for tag, count in Counter(examples[:, -1]).iteritems()]
    info_gain = "info-gain"
    s_entropy = entropy(probabilietes) if gain_measure == info_gain else min(probabilietes)
    values_count = count_values(attributes, examples)
    attributes_information_gain = {}
    classifiers = 'yesAndNoTags'
    for (attribute, value) in values_count:
        handleAttributeAndValue(attribute, attributes_information_gain, classifiers, examples, gain_measure, info_gain,
                                s_entropy, value, values_count)

    return returnMaxAttribute(attributes_information_gain)


def handleAttributeAndValue(attribute, attributes_information_gain, classifiers, examples, gain_measure, info_gain,
                            s_entropy, value, values_count):
    if attribute not in attributes_information_gain:
        attributes_information_gain[attribute] = s_entropy
    probabilietes = []
    tags_and_counts = values_count[(attribute, value)][('%s' % classifiers)]
    for tag, count in tags_and_counts.iteritems():
        probabilietes.append(float(count) / values_count[(attribute, value)]['count'])
    initTempEntropy(attribute, attributes_information_gain, examples, gain_measure, info_gain, probabilietes, value,
                    values_count)


class Node:
    def __init__(self, node_attribute, value):
        self.node_value = value
        self.node_attribute = node_attribute
        self.node_sub_tree = []

    def insert_sub_tree(self, sub_tree):
        self.node_sub_tree.append(sub_tree)

    def get_node_attribute(self):
        return self.node_attribute

    def get_node_sub_trees(self):
        return self.node_sub_tree

    def get_value(self):
        return self.node_value

    def is_have_sub_trees(self):
        return len(self.node_sub_tree) > 0

def initTempEntropy(attribute, attributes_information_gain, examples, gain_measure, info_gain, probabilietes, value,
                    values_count):
    if gain_measure == info_gain:
        temp_entropy = entropy(probabilietes)
    else:
        temp_entropy = min(probabilietes)
    attributes_information_gain[attribute] -= float(values_count[(attribute, value)]['count']) / len(
        examples) * temp_entropy


def returnMaxAttribute(attributes_information_gain):
    max_value = -1.0
    best_attribute = None
    for attribute_and_val in attributes_information_gain.iteritems():
        if attribute_and_val[1] > max_value:
            max_value = attribute_and_val[1]
            best_attribute = attribute_and_val[0]
    return best_attribute


def calculate_c(taggings_counter, total_size, gain_measure):
    probs = [float(count) / total_size for tag, count in taggings_counter.iteritems()]
    return entropy(probs) if gain_measure == "info-gain" else min(probs)


def count_values(attributes, examples):
    attribute_and_value_to_counts = {}
    for example in examples:
        for idx, value in enumerate(example[:-1]):
            attribute = attributes[:-1][idx]
            if (attribute, value) not in attribute_and_value_to_counts:
                attribute_and_value_to_counts[(attribute, value)] = {'count': 0, 'yesAndNoTags': Counter()}
            attribute_and_value_to_counts[(attribute, value)]['count'] += 1
            attribute_and_value_to_counts[(attribute, value)]['yesAndNoTags'][example[-1]] += 1

    return attribute_and_value_to_counts


def entropy(probablities):
    return -1 * sum(prob * log(prob, 2) for prob in probablities)


def predict(tree, example, attributes):
    while tree.is_have_sub_trees():
        value = example[attributes.index(tree.node_attribute)]
        tree = list(child for child in tree.get_node_sub_trees() if child.value == value)[0]
        if not tree.is_have_sub_trees():
            return tree.node_attribute


def calc_accuracy(attributes, examples, tree):
    correct = 0
    for idx, example in enumerate(examples):
        prediction = predict(tree, example, attributes)
        correct += 1 if prediction == example[-1] else 0
        print '%s: %s' % (i, prediction)
    return float(correct) / len(examples)


def parse_validation_file(validation_file_path):
    text = np.loadtxt(validation_file_path, dtype=str)
    attributes = text[0].tolist()
    examples = text[1:]

    return attributes, examples

if __name__ == '__main__':
    train_file_path = sys.argv[1]
    validation_file_path = sys.argv[2]
    measure_gain = sys.argv[3]

    attributes_to_values, attributes, examples_list = parse_train_file(train_file_path)

    tree = id3_algorithm(examples_list, attributes, attributes_to_values, "info-gain")
    output_file = open("output.txt", 'w')
    write_final_tree(tree,0, output_file)

    attributes, examples = parse_validation_file(validation_file_path)
    # accuracy = calc_accuracy(attributes, examples, tree)

    print accuracy