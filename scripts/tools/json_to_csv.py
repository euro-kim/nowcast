import pandas as pd

# Read the JSON file
df = pd.read_json(r'c:\Users\home\Documents\Git\nowcast\assets\data.json')

# Save as CSV
df.to_csv(r'c:\Users\home\Documents\Git\nowcast\assets\data.csv', index=False)
