# Descarga desde UCIMLREPO
from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
darwin = fetch_ucirepo(id=732) 
  
# data (as pandas dataframes) 
x = darwin.data.features 
y = darwin.data.targets 
  
# metadata 
print(darwin.metadata) 
  
# variable information 
print(darwin.variables) 
