# Import package.

import fuzzylite as fl
#import numpy as np
#import skfuzzy as fuzz
#import matplotlib.pyplot as plt

# Declaring and Initializing the fuzzy engine
engine=fl.Engine(
    name="SimpleDimmer",
    description="Simple Dimmer Fuzzy System which dims light based upon Light Conditions"
)
    
# Defining the Input Variables (Fuzzification)
engine.input_Variables=[
    fl.InputVariable(
        name="Ambient",
        description="",
        enabled=True,
        minimum=0.000,
        maximum=1.000,
        lock_range=False,
        terms=[
            fl.Triangle("DARK", 0.000, 0.250, 0.500), # Triangular MF defining DARK
            fl.Triangle("MEDIUM", 0.250, 0.500, 0.750), # Triangular MF defining MEDIUM
            fl.Triangle("BRIGHT", 0.500, 0.750, 1.000), # Triangular MF defining BRIGHT
        ]
    )
]

# Defining the Output Variables (Defuzzificztion)
engine.output_Variables=[
    fl.OutputVariable(
        name="Power",
        description="",
        enabled=True,
        minimum=0.000,
        maximum=1.000,
        lock_range=False,
        aggregation=fl.Maximum(),
        defuzzifier=fl.Centroid(200),
        lock_previous=False,
        terms=[
            fl.Triangle("LOW", 0.000, 0.250, 0.500), # Triangular MF defining LOW Light
            fl.Triangle("MEDIUM", 0.250, 0.500, 0.750), # Triangular MF defining MEDIUM Light
            fl.Triangle("HIGH", 0.500, 0.550, 1.000), # Triangular MF defining HIGH Light
        ]
    )
]

# Creation of Fuzzy Rule Base
engine.rule_blocks=[
    fl.RuleBlock(
        name="",
        description="",
        enabled=True,
        conjunction=None,
        disjunction=None,
        implication=fl.Minimum(),
        activation=fl.General(),
        rules=[
            fl.Rule.create("if Ambient is DARK then Power is HIGH", engine),
            fl.Rule.create("if Ambient is MEDIUM then Power is MEDIUM", engine),
            fl.Rule.create("if Ambient is BRIGHT then Power is LOW", engine)
        ]
    )
]