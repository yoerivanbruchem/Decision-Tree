import math
from collections import Counter
import pydot
import pandas as pd


class DecisionTree:
    """Class for creating a Decision Tree."""

    def __init__(self, data, features, resulting_feature):
        """Create an instance of the decision tree."""
        self.__train_data = data
        self.__features = features
        self.__resulting_feature = resulting_feature
        self.__tree = self.__built_tree(self.__train_data, self.__features, self.__resulting_feature, None)
        self.accuracy_of_previous_test = 0

    def __calculate_frequency(self, data_set, feature):
        """Return the determined frequency of each value in the data set for a given feature."""
        feature_data = list(data_set[feature])
        value_frequency = Counter(feature_data)
        return dict(value_frequency)

    def __entropy(self, data_set, target_feature):
        """Return the calculated the entropy of the target_feature in the given data set."""
        frequencies = self.__calculate_frequency(data_set, target_feature)
        feature_entropy = 0.0
        number_of_values = len(data_set)

        # Add entropy for each value in frequencies.
        for frequency in frequencies:
            probability = frequencies[frequency] / number_of_values
            feature_entropy += (probability * math.log(probability, 2))

        return feature_entropy * -1

    def __gain(self, data_set, split_feature, target_feature):
        """Return the calculated information gain for a given split_feature to another target_feature."""
        frequencies = self.__calculate_frequency(data_set, split_feature)
        data_entropy = 0.0

        # Calculate the entropy of the data.
        for value, frequency in frequencies.items():
            probability = frequency / sum(frequencies.values())
            data_subset = data_set[data_set[split_feature] == value]
            data_entropy += probability * self.__entropy(data_subset, target_feature)

        return self.__entropy(data_set, target_feature) - data_entropy

    def __built_tree(self, data_set, features, target_feature, default_class):
        """Built a tree using the data_set and the given features."""
        tree_features = features[:]
        data_set = data_set[:]
        result_class = Counter(x for x in data_set[target_feature])
        result_class_count = len(result_class)
        result_class_maximum_value = result_class.most_common(1)[0][0]

        # This branch is a leaf (all the results belong to the same class)
        if result_class_count == 1:
            result = list(result_class.keys())
            return result[0]

        # Check if the data set is empty or the attributes are not given.
        elif data_set.empty or (not tree_features):
            return default_class

        else:
            # Get default value for next branch.
            default_class = result_class_maximum_value

            # Get split feature.
            feature_gains = {}
            for feature in tree_features:
                feature_gains[feature] = self.__gain(data_set, feature, target_feature)

            split_feature = max(feature_gains, key=feature_gains.get)
            tree = {split_feature: {}}

            # Remove current feature from feature list.
            remaining_features = tree_features
            remaining_features.remove(split_feature)

            # Create a subtree for each child feature
            for feature_value, data_subset in data_set.groupby(split_feature):
                subtree = self.__built_tree(data_subset, remaining_features, target_feature, default_class)
                tree[split_feature][feature_value] = subtree

            return tree

    def get_tree(self):
        """Return the tree."""
        return self.__tree

    def __classify(self, instance, tree, default=None):
        """Classify a given instance using the trained tree."""
        attribute = str(list(tree.keys())[0])
        keys_of_attribute = list(tree[attribute].keys())
        if instance[attribute].iloc[0] in keys_of_attribute:
            subtree = tree[attribute]
            result = subtree[instance[attribute].iloc[0]]
            if isinstance(result, dict):
                return self.__classify(instance, result)
            else:
                return result
        else:
            return default

    def classify_data(self, test_set, include_features_in_result=False):
        """Classify a data test set."""
        if len(test_set) == 1:
            return self.__classify(test_set, self.__tree)
        else:

            indices = test_set.index.values.tolist()
            correct_classified_rows = 0

            classification_result = []

            for index in indices:

                training_row = pd.DataFrame(test_set.loc[index])
                training_row = training_row.T

                result_row = [list(x) for x in training_row.values][0]
                expected_value = str(training_row[self.__resulting_feature].iloc[0])
                classified_value = self.classify_data(training_row)
                result_row.append(classified_value)
                result_row = tuple(result_row)

                classification_result.append(result_row)

                if expected_value == classified_value:
                    correct_classified_rows += 1

            self.accuracy_of_previous_test = (correct_classified_rows / len(test_set) * 100)

            column_names = list(test_set)
            column_names.append("classified")
            classification_result = pd.DataFrame(classification_result, columns=column_names)

            if include_features_in_result:
                return classification_result
            else:
                return classification_result.iloc[:, -2:]
