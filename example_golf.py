import pandas as pd
import decision_tree

# Get data.
features = ["Outlook", "Temperature", "Humidity", "Windy"]
data = pd.DataFrame(pd.read_csv('golf.csv'))

# Create tree.
tree = decision_tree.Tree(data, features, "Result")

# Create visualization.
file_name = "golf"
tree.create_visualization(file_name)

# Classify data.
conditions = {"Outlook": "sunny", "Temperature": "hot", "Humidity": "high", "Windy": "strong"}
conditions = pd.DataFrame(conditions, index=[0])
print(tree.classify_data(conditions))

from_data_set = data.loc[:0]
print(tree.classify_data(from_data_set))

# Classify accuracy
tree.split_data(20)
print(tree.calculate_accuracy())
