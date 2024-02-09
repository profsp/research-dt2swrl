#**************************************************
# Test script for exploring the package capabilities
#**************************************************

from owlready2 import *
import dt2swrl as ds
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import numpy as np # linear algebra

# load the ontology data to a dataframe
data = ds.owl_to_dataframe(r"C:\...\git_sleep_ontology.rdf")


#prepare the dataframe for the decision tree
data['Gender'] = data['Gender'].replace("Male", 0)
data['Gender'] = data['Gender'].replace("Female", 1)
data['SleepDisorderAttr'] = data['SleepDisorderAttr'].replace("No", 0)
data['SleepDisorderAttr'] = data['SleepDisorderAttr'].replace("Sleep Apnea", 1)
data['SleepDisorderAttr'] = data['SleepDisorderAttr'].replace("Insomnia", 2)

features = data.drop(columns=['SleepDisorderAttr', 'Name'])
labels = data['SleepDisorderAttr']

# apply decision tree learning
f_train, f_test, l_train, l_test = train_test_split(features, labels, train_size = 0.8, random_state = None,shuffle=False)

cfl = DecisionTreeClassifier(random_state=42)

cfl.fit(f_train, l_train)

feature_cols=['Age','BMICategory', 'BloodPressure', 'DailySteps', 'Gender',  'HeartRate', 'PhysicalActivityLevel',	'QualityofSleep'	,'SleepDuration'	, 'StressLevel']

# specify SWRL-like classes to be used in the ontology
class_names = ['No(?x)', 'Sleep Apnea(?x)', 'Insomnia(?x)']

# create swrl_rules from dt knowledge
swrl_rules = ds.getSWRLRules(cfl, class_names, feature_cols)

# add swrl rules to ontology for reasoning 
ds.save_dt_to_owl(r"C:\...\git_sleep_ontology.rdf", swrl_rules)

