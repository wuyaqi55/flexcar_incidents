import pandas as pd
from judger import fault_prediction_gpt1 as j

data = pd.read_csv("../data/incidents_descript.csv")
data = data.head(20)
fault = []
for i in data['What Happened']:
    fault.append(j(i))
data['fault'] = fault
data.to_csv('yes')