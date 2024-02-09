#!/usr/bin/python

from sklearn.tree import _tree
import numpy as np
from owlready2 import *
import pandas as pd


#******************************************************************************************
# retrieve individuals incl data property values from an ontology and transform them into a dataframe
#******************************************************************************************
def owl_to_dataframe(owl_filepath):
    onto = get_ontology(owl_filepath).load()
    individuals = list(onto.individuals())
    data_properties = list(onto.data_properties())
    data = []

    for individual in individuals:
        individual_data = {}
        individual_data["Name"] = individual.name
        for data_prop in data_properties:
            values = individual.get_properties()
            if values:
                individual_data[data_prop.name] = getattr(individual, data_prop.name)[0]

        data.append(individual_data)

    df = pd.DataFrame(data)
    return df

#*****************************************************************************************
# extract the rules in if-then-else
#*****************************************************************************************
def get_rules(tree, feature_names, class_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    paths = []
    path = []
    
    def recurse(node, path, paths):
 
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            p1, p2 = list(path), list(path)
            p1 += [f"({name} <= {np.round(threshold, 3)})"]
            recurse(tree_.children_left[node], p1, paths)
            p2 += [f"({name} > {np.round(threshold, 3)})"]
            recurse(tree_.children_right[node], p2, paths)
        else:
            path += [(tree_.value[node], tree_.n_node_samples[node])]
            paths += [path]
            
    recurse(0, path, paths)

    samples_count = [p[-1][1] for p in paths]
    ii = list(np.argsort(samples_count))
    paths = [paths[i] for i in reversed(ii)]
    
    rules = []
    rulesFormated = []
    for path in paths:
        rule = "if "
        
        for p in path[:-1]:
            if rule != "if ":
                rule += " and "
            rule += str(p)
        rule += " then "
        if class_names is None:
            rule += "response: "+str(np.round(path[-1][0][0][0],3))
        else:
            classes = path[-1][0][0]
            l = np.argmax(classes)
            rule += f"{class_names[l]}" 
        rules += [rule]
        rulesFormated.append(f''+rule + "\n")
   
    #print('if-then-else count: ' + str(len(rules)))
    return rules
    

#*****************************************************************************************
#transform if-then-else-rules into swrl rules
#*****************************************************************************************
def getSWRLRules(cfl, class_names, feature_cols):
    raw_rules = get_rules(cfl, feature_cols, class_names)
    swrl_rules = []
    for line in raw_rules:
        
            rule_parts = line.split('then')
            antecedent = rule_parts[0].strip().replace('if', '')
            antecedent_parts = antecedent.split('and')
            
            for p in antecedent_parts:
                if len(p)==0:
                    antecedent_parts.remove(p)
            
            antecedent_swrl = ""
            for i in range(len(antecedent_parts)):
                antecedent_parts[i] = antecedent_parts[i].replace('(', '').replace(')', '').replace('and', ', ').strip()
                if '<=' in antecedent_parts[i]:
                    ops = antecedent_parts[i].split('<=')
                    antecedent_parts[i] = ops[0].strip().replace(' ', '') + "(?x,?y"+str(i)+") ^ " + "lessThanOrEqual(?y"+str(i)+", "+ops[1].strip().replace(' ', '') + ")"
                    if len(antecedent_swrl) > 0: 
                        antecedent_swrl = antecedent_swrl + " ^ " + antecedent_parts[i]
                    else: 
                        antecedent_swrl = antecedent_parts[i]
                elif '> ' in antecedent_parts[i]:
                    ops = antecedent_parts[i].split('>')
                    antecedent_parts[i] = ops[0].strip().replace(' ', '') + "(?x,?y"+str(i)+") ^ " + "greaterThan(?y"+str(i)+", "+ops[1].strip().replace(' ', '') + ")"
                    if len(antecedent_swrl) > 0: 
                        antecedent_swrl = antecedent_swrl + " ^ " + antecedent_parts[i]
                    else: 
                        antecedent_swrl = antecedent_parts[i]
                elif '< ' in antecedent_parts[i]:
                    ops = antecedent_parts[i].split('<')
                    antecedent_parts[i] = ops[0].strip().replace(' ', '') + "(?x,?y"+str(i)+") ^ " + "lessThan(?y"+str(i)+", "+ops[1].strip().replace(' ', '') + ")"
                    if len(antecedent_swrl) > 0: 
                        antecedent_swrl = antecedent_swrl + " ^ " + antecedent_parts[i]
                    else: 
                        antecedent_swrl = antecedent_parts[i]
                elif '>=' in antecedent_parts[i]:
                    ops = antecedent_parts[i].split('>=')
                    antecedent_parts[i] = ops[0].strip().replace(' ', '') + "(?x,?y"+str(i)+") ^ " + "greaterThanOrEqual(?y"+str(i)+", "+ops[1].strip().replace(' ', '') + ")"
                    if len(antecedent_swrl) > 0: 
                        antecedent_swrl = antecedent_swrl + " ^ " + antecedent_parts[i]
                    else: 
                        antecedent_swrl = antecedent_parts[i]


                consequent_swrl = rule_parts[1].strip().replace(' ','')

            swrl_rule = f"{antecedent_swrl} -> {consequent_swrl}"
            swrl_rules.append(swrl_rule)
 
    #print('swrl count: ' + str(len(swrl_rules)))
    return swrl_rules 


#*****************************************************************************************
# add newly created SWRL rules to the ontology and persist them in the RDF file
#*****************************************************************************************
def save_dt_to_owl(owl_filepath, swrl_rules):
    onto = get_ontology(owl_filepath).load()
    
    with onto:
       
        i=1
        for p in swrl_rules:
            rule = Imp()
            rule.label.append("DT_GEN_"+str(i))
            temp = """%""".replace("%", p)
            rule.set_as_rule(temp)
            
            i=i+1

    onto.save(file=owl_filepath, format="rdfxml")