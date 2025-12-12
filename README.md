# Temporal-Diverse-Club

Our repository includes three datasets of varying scales. 
We also paste our original data in the repository.
We summarize the contributions and distinctions of the following three datasets.
----------neuroscience.py----------
participation coefficient
clubness
machine learning
edge density (supplementary materials)

----------airline.py----------
temporal participation coefficient
clubness
three traditional metrics
kumramoto model
clique detection (supplementary materials)

----------ant.py----------
weighted participation coefficient
community algorithm to align labels
strong vs weak networks
club transition

----------Some important parameters----------
dy_mat: real matrices np.adarray T*N*N
dy_gra: real graphs T*G
dy_ran: null network np.adarray T*N*N
agg: aggregated matrix
AGG: aggregated graph
G means graph and g means matrix
You need to notice the difference between uppercase and lowercase!

----------Some important packages----------
networkx 3.3
pandas 2.2.3
community 1.1.0
infomap 1.0.1

