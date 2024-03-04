import time
start_time = time.time()
import pandas as pd
import numpy as np
# OBS
# The structure needs to be 0 = intensity, 1 = x/y and 2 = y/x

host = 'T114/W72/'
# Load data from CSV files (assuming CSV format)
tableA = pd.read_csv(f'{host}EVs 385nm.csv' , usecols=['IntensitySum1_VioBlue__IntensitySumOfChannel_VioBlue___R','CenterX__CenterX__R','CenterY__CenterY__R'], header=0)
tableB = pd.read_csv(f'{host}EVs 475nm.csv', usecols=['IntensitySum1_FITC__IntensitySumOfChannel_FITC___R','CenterX__CenterX__R','CenterY__CenterY__R'], header=0)
tableC = pd.read_csv(f'{host}EVs 555nm.csv', usecols=['IntensitySum1_RPe__IntensitySumOfChannel_RPe___R','CenterX__CenterX__R','CenterY__CenterY__R'], header=0)
tableD = pd.read_csv(f'{host}EVs 630nm.csv', usecols=['IntensitySum1_APC__IntensitySumOfChannel_APC___R','CenterX__CenterX__R','CenterY__CenterY__R'], header=0)
tableA = pd.read_csv(f'{host}EVs 475nm.csv', usecols=['IntensitySum1_FITC__IntensitySumOfChannel_FITC___R','CenterX__CenterX__R','CenterY__CenterY__R'], header=0)

# Calculate lengths of dataframes and zip with dfs

df = [tableA, tableB, tableC, tableD]

lengths_dfs = [(len(df), df) for df in df]

# Sort the list of tuples based on the length in descending order
sorted_lengths_dfs = sorted(lengths_dfs, key=lambda x: x[0], reverse=True)

# Extract the sorted dataframes from the sorted list of tuples
sdf = [df for _, df in sorted_lengths_dfs]
del df, sorted_lengths_dfs, tableA, tableB, tableC, tableD
# Convert pandas DataFrames to NumPy arrays
tableA_np = sdf[0].to_numpy()
tableB_np = sdf[1].to_numpy()
tableC_np = sdf[2].to_numpy()
tableD_np = sdf[3].to_numpy()


coloc_table = np.zeros((0, 6))  # Initialize coloc_table as empty numpy array

print(len(tableA_np)+len(tableB_np)+len(tableC_np)+len(tableD_np))

d = 0.6  # Global value for comparison


def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

# Return true if there is a smaller distance
def precheck_distance(point1,col,d):
    dist = np.linalg.norm(col - point1, axis=1)
    return any(dist < d)

def shortest_distance(point1,col,d):
    dist = np.linalg.norm(col - point1, axis=1)
    closest_index = np.argmin(dist)
    return closest_index

#€€
# Start with the first table 
for row_a in tableA_np:
    new_row = np.array([row_a[1], row_a[2], row_a[0], 0, 0, 0]).reshape(1, -1)
    coloc_table = np.concatenate([coloc_table, new_row], axis=0)      
    # Compare with table B
    if precheck_distance(row_a[1:],tableB_np[:,1:],d):
        closest_index = shortest_distance(row_a[1:],tableB_np[:,1:],d)
        coloc_table[-1,3] = tableB_np[closest_index,0]    
        tableB_np = np.delete(tableB_np,closest_index, axis=0)    
    # Compare with table C
    if precheck_distance(row_a[1:],tableC_np[:,1:],d):
        closest_index = shortest_distance(row_a[1:],tableC_np[:,1:],d)
        coloc_table[-1,4] = tableC_np[closest_index,0]    
        tableC_np = np.delete(tableC_np,closest_index, axis=0) 
    # Compare with table D
    if precheck_distance(row_a[1:],tableD_np[:,1:],d):
        closest_index = shortest_distance(row_a[1:],tableD_np[:,1:],d)
        coloc_table[-1,5] = tableD_np[closest_index,0]    
        tableD_np = np.delete(tableD_np,closest_index, axis=0) 
    tableA_np = np.delete(tableA_np,0, axis=0) 
print('Table A done')

# Then the next table (B)
for row_b in tableB_np:
    new_row = np.array([row_b[1], row_b[2], 0, row_b[0], 0, 0]).reshape(1, -1)
    coloc_table = np.concatenate([coloc_table, new_row], axis=0)         
    # Compare with table C
    if precheck_distance(row_b[1:],tableC_np[:,1:],d):
        closest_index = shortest_distance(row_b[1:],tableC_np[:,1:],d)
        coloc_table[-1,4] = tableC_np[closest_index,0]    
        tableC_np = np.delete(tableC_np,closest_index, axis=0) 
    # Compare with table D
    if precheck_distance(row_b[1:],tableD_np[:,1:],d):
        closest_index = shortest_distance(row_b[1:],tableD_np[:,1:],d)
        coloc_table[-1,5] = tableD_np[closest_index,0]    
        tableD_np = np.delete(tableD_np,closest_index, axis=0) 
    tableB_np = np.delete(tableB_np,0, axis=0) 
print('Table B done')

# Then the next table (C)
for row_c in tableC_np:
    new_row = np.array([row_c[1], row_c[2], 0, 0, row_c[0], 0]).reshape(1, -1)
    coloc_table = np.concatenate([coloc_table, new_row], axis=0)         
    # Compare with table D
    if precheck_distance(row_c[1:],tableD_np[:,1:],d):
        closest_index = shortest_distance(row_c[1:],tableD_np[:,1:],d)
        coloc_table[-1,5] = tableD_np[closest_index,0]    
        tableD_np = np.delete(tableD_np,closest_index, axis=0) 
    tableC_np = np.delete(tableC_np,0, axis=0) 

print('Table C done')

# Lastly, table D
for row_d in tableD_np:
    new_row = np.array([row_d[1], row_d[2], 0, 0, 0, row_d[0]]).reshape(1, -1)
    coloc_table = np.concatenate([coloc_table, new_row], axis=0)
    tableD_np = np.delete(tableD_np,0, axis=0) 

print('Table D done')

# if you are only doing colocalization amongst some
coloc_table = coloc_table[~np.all(coloc_table == 0, axis=1)]

coloc_table = coloc_table[(coloc_table == 0).sum(axis=1).argsort()]

columns=['Center X', 'Center Y', f'{sdf[0].columns[0]}', f'{sdf[1].columns[0]}', f'{sdf[2].columns[0]}', f'{sdf[3].columns[0]}']

coloc_table = pd.DataFrame(coloc_table,columns=columns)
coloc_table = coloc_table.T.drop_duplicates().T
#coloc_table.to_csv(f'{host}colocalized_all.csv', index=False) 
print("--- %s seconds ---" % (time.time() - start_time)) 
