import fuzzylite as fl

#Declaring and Initializing the Fuzzy Engine
engine = fl.Engine(
      name="SimpleDimmer",
      description="Simple Dimmer Fuzzy System which dims light based upon Light Conditions"
)

#Defining the Input Variables (Fuzzification)
engine.input_variables = [
      fl.InputVariable(
      name="Ambient",
      description="",
      enabled=True,
      minimum=0.000,
      maximum=1.000,
      lock_range=False,
      terms=[
      fl.Triangle("DARK", 0.000, 0.250, 0.500), #Triangular Membership Function defining "Dark"
      fl.Triangle("MEDIUM", 0.250, 0.500, 0.750), #Triangular Membership Function defining "Medium"
      fl.Triangle("BRIGHT", 0.500, 0.750, 1.000) #Triangular Membership Function defining "Bright"
        ]
    )
]

#Defining the Output Variables (Defuzzification)
engine.output_variables = [
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
      fl.Triangle("LOW", 0.000, 0.250, 0.500), #Triangular Membership Function defining "LOW Light"
      fl.Triangle("MEDIUM", 0.250, 0.500, 0.750), #Triangular Membership Function defining "MEDIUM light"
      fl.Triangle("HIGH", 0.500, 0.750, 1.000) #Triangular Membership Function defining "HIGH Light"
        ]
    )
]

#Creation of Fuzzy Rule Base
engine.rule_blocks = [
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
