import numpy as np
import pandas as pd
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import BayesianModel
from pgmpy.inference import VariableElimination 
from pgmpy.estimators import MaximumLikelihoodEstimator, BayesianEstimator

class bayesian_network:
    def __init__(self):
        pass

    def model(self):
        student_model = BayesianModel([('D', 'G'), ('I', 'G'), ('G', 'L'), ('I', 'S')])
        grade_cpd = TabularCPD(
        variable='G', # node name
        variable_card=3, # number of variable in this node
        values=[[0.3, 0.05, 0.9, 0.5], # probability of node
        [0.4, 0.25, 0.08, 0.3],
        [0.3, 0.7, 0.02, 0.2]],
        evidence=['I', 'D'], # parent node
        evidence_card=[2, 2] # number of variable in each parent node
        )

        difficulty_cpd = TabularCPD(
                    variable='D',
                    variable_card=2,
                    values=[[0.6], [0.4]]
        )

        intel_cpd = TabularCPD(
                    variable='I',
                    variable_card=2,
                    values=[[0.7], [0.3]]
        )

        letter_cpd = TabularCPD(
                    variable='L',
                    variable_card=2,
                    values=[[0.1, 0.4, 0.99],
                    [0.9, 0.6, 0.01]],
                    evidence=['G'],
                    evidence_card=[3]
        )

        sat_cpd = TabularCPD(
                    variable='S',
                    variable_card=2,
                    values=[[0.95, 0.2],
                    [0.05, 0.8]],
                    evidence=['I'],
                    evidence_card=[2]
        )

        student_model.add_cpds(grade_cpd, difficulty_cpd, intel_cpd, letter_cpd, sat_cpd)
        
        student_infer = VariableElimination(student_model)
        prob_G = student_infer.query( variables=['G'], evidence={'I': 1, 'D': 0}) 
        print(prob_G)
        
        return student_model

    def fit(self, model, data):
        model.fit(data, estimator=MaximumLikelihoodEstimator)
        for cpd in model.get_cpds():
            print("CPD of {variable}:".format(variable=cpd.variable)) 
            print(cpd)

if __name__=="__main__":
    bn = bayesian_network()
    raw_data = np.random.randint(low=0, high=2, size=(1000, 5))
    data = pd.DataFrame(raw_data, columns=['D', 'I', 'G', 'L', 'S'])
    student_model = bn.model()
    bn.fit(student_model, data)