import pandas as pd

states = [
"AL","AK","AZ","AR","CA","CO","CT","DE","DC","FL","GA","HI","ID","IL","IN",
"IA","KS","KY","LA","ME","MD","MA","MI","MN","MS","MO","MT","NE","NV","NH",
"NJ","NM","NY","NC","ND","OH","OK","OR","PA","RI","SC","SD","TN","TX","UT",
"VT","VA","WA","WV","WI","WY","PR"
]

neighbors = {
"AL":["FL","GA","MS","TN"],
"AK":["WA"],
"AZ":["CA","NV","UT","NM","CO"],
"AR":["MO","TN","MS","LA","TX","OK"],
"CA":["OR","NV","AZ","HI"],
"CO":["WY","NE","KS","OK","NM","AZ","UT"],
"CT":["NY","MA","RI"],
"DE":["MD","NJ","PA"],
"DC":["MD","VA"],
"FL":["GA","AL","PR"],
"GA":["FL","AL","TN","NC","SC"],
"HI":["CA"],
"ID":["WA","OR","NV","UT","WY","MT"],
"IL":["WI","IA","MO","KY","IN","MI"],
"IN":["MI","OH","KY","IL"],
"IA":["MN","SD","NE","MO","IL","WI"],
"KS":["NE","MO","OK","CO"],
"KY":["IL","IN","OH","WV","VA","TN","MO"],
"LA":["TX","AR","MS"],
"ME":["NH"],
"MD":["VA","WV","PA","DE","DC"],
"MA":["RI","CT","NY","VT","NH"],
"MI":["WI","IN","OH","IL"],
"MN":["WI","IA","SD","ND"],
"MS":["LA","AR","TN","AL"],
"MO":["IA","IL","KY","TN","AR","OK","KS","NE"],
"MT":["ID","WY","SD","ND"],
"NE":["SD","IA","MO","KS","CO","WY"],
"NV":["CA","OR","ID","UT","AZ"],
"NH":["ME","MA","VT"],
"NJ":["NY","PA","DE"],
"NM":["AZ","UT","CO","OK","TX"],
"NY":["PA","NJ","CT","MA","VT"],
"NC":["VA","TN","GA","SC"],
"ND":["MT","SD","MN"],
"OH":["PA","WV","KY","IN","MI"],
"OK":["KS","MO","AR","TX","NM","CO"],
"OR":["WA","ID","NV","CA"],
"PA":["NY","NJ","DE","MD","WV","OH"],
"RI":["CT","MA"],
"SC":["GA","NC"],
"SD":["ND","MN","IA","NE","WY","MT"],
"TN":["KY","VA","NC","GA","AL","MS","AR","MO"],
"TX":["NM","OK","AR","LA"],
"UT":["ID","WY","CO","NM","AZ","NV"],
"VT":["NY","NH","MA"],
"VA":["NC","TN","KY","WV","MD","DC"],
"WA":["OR","ID","AK"],
"WV":["OH","PA","MD","VA","KY"],
"WI":["MN","IA","IL","MI"],
"WY":["MT","SD","NE","CO","UT","ID"],
"PR":["FL"]
}

# build adjacency
A = pd.DataFrame(0, index=states, columns=states)

for s, neighs in neighbors.items():
    for n in neighs:
        A.loc[s, n] = 1
        A.loc[n, s] = 1   # enforce symmetry

A.to_csv('adjacency_matrix.csv', index=True)