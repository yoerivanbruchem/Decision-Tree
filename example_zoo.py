import pandas as pd
import decision_tree as dt
import data_tool_set as dts

data = pd.DataFrame(pd.read_csv('zoo.csv'))
data = pd.DataFrame(data.drop('animal_name', 1))
features = list(data.columns.values)
features.remove('class_type')

# Prepare Data
data['hair'] = data['hair'].map({1: "Yes", 0: "No"})
data['feathers'] = data['feathers'].map({1: "Yes", 0: "No"})
data['milk'] = data['milk'].map({1: "Yes", 0: "No"})
data['eggs'] = data['eggs'].map({1: "Yes", 0: "No"})
data['airborne'] = data['airborne'].map({1: "Yes", 0: "No"})
data['aquatic'] = data['aquatic'].map({1: "Yes", 0: "No"})
data['predator'] = data['predator'].map({1: "Yes", 0: "No"})
data['toothed'] = data['toothed'].map({1: "Yes", 0: "No"})
data['backbone'] = data['backbone'].map({1: "Yes", 0: "No"})
data['breathes'] = data['breathes'].map({1: "Yes", 0: "No"})
data['venomous'] = data['venomous'].map({1: "Yes", 0: "No"})
data['fins'] = data['fins'].map({1: "Yes", 0: "No"})
data['tail'] = data['tail'].map({1: "Yes", 0: "No"})
data['domestic'] = data['domestic'].map({1: "Yes", 0: "No"})
data['catsize'] = data['catsize'].map({1: "Yes", 0: "No"})
data['class_type'] = data['class_type'].map({1: "Mammal", 2: "Bird", 3: "Reptile", 4: "Fish", 5:
                                            "Amphibian", 6: "Bug", 7: "Invertebrate"})

# Create Tree
tree = dt.Tree(data, features, "class_type")

# Visualize Tree
tree.create_visualization('zoo_from_total_set')

# Classify single (new) record.
animal = {'hair': 'Yes', 'feathers': 'No', 'eggs': 'No', 'milk': 'Yes', 'airborne': 'No', 'aquatic': 'No',
          'predator': 'Yes','toothed': 'Yes', 'backbone': 'Yes', 'breathes': 'Yes', 'venomous': 'No', 'fins': 'No',
          'legs': 4, 'tail': 'Yes','domestic': 'No', 'catsize': 'No'}

animal = pd.DataFrame(animal, index=[0])
print("Species:", tree.classify_data(animal))

# Classify test_set
data_tools = dts.DataToolSet(data)
data_tools.split_data(66)

tree = dt.Tree(data_tools.train_data, features, "class_type")
tree.create_visualization("zoo_from_training_set")

print(tree.classify_data(data_tools.test_data))
print("Accuracy: " + str(tree.accuracy_of_previous_test))
