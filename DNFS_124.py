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
      fl.Bell("DARK", -10.000, 5.000, 3.000), #Generalized Bell Membership Function defining "Dark"
      fl.Bell("MEDIUM", 0.000, 5.000, 3.000), #Generalized Bell  Membership Function defining "Medium"
      fl.Bell("BRIGHT", 10.000, 5.000, 3.000) #Generalized Bell  Membership Function defining "Bright"
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
      fl.Sigmoid("LOW", 0.500, -30.000), #Triangular Membership Function defining "LOW Light"
      fl.Sigmoid("MEDIUM", 0.130, 30.000), #Triangular Membership Function defining "MEDIUM light"
      fl.Sigmoid("HIGH", 0.830, 30.000), #Triangular Membership Function defining "HIGH Light"
      fl.Triangle("BRIGHT", 0.500, 0.750, 1.000) #Triangular Membership Function defining "Bright"
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
      implication=None,
      activation=fl.General(),
      rules=[
      fl.Rule.create("if Ambient is DARK then Power is HIGH", engine),
      fl.Rule.create("if Ambient is MEDIUM then Power is MEDIUM", engine),
      fl.Rule.create("if Ambient is BRIGHT then Power is LOW", engine)
        ]
      )
]
