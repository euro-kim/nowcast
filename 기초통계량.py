import json
import pandas as pd

with open('assets/data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

df = pd.DataFrame(data)

print(df.describe(include='all'))  
