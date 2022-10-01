###### IMPORTANT
# The code is made to run in the ADNI folder. 
# The files Diagnosis and MR_Image_Analysis_ADNI_1_2_GO_FS51 should be unzip
# The final data is not stored. 
#
# Final data shape
# data = contains only the TA information   - ndarray [participants, features]
# Gender = Contains the patients gender     - ndarray [participants,]
# Age = Contains the age of the patients    - ndarray [participants,]
# Site = Contains the site information      - ndarray [participants,]

### Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
 
def get_uniques_samples(data):
    # Check the different date format
    keys = data.keys()

    if "USERDATE" in keys :
        date_name = "USERDATE"
    elif "EXAMDATE" in keys:
        date_name = "EXAMDATE"

    # Keep only the oldes image for each RID
    for num, RID in enumerate(np.unique(data["RID"])):
        data_aux = data[data["RID"]==RID]

        id_min = np.min(data_aux[date_name])

        data_aux = data_aux[data_aux[date_name]==id_min].iloc[0]

        if num == 0:
            data_return = data_aux
        else:
            data_return = np.vstack([data_return,data_aux])

    # For a DataFrame Again with the same keys.
    data_return = pd.DataFrame(data_return, columns = keys)

    return data_return


################################################# Diagnostic filter
root_dir = "Diagnosis/"

data_name = "DXSUM_PDXCONV_ADNIALL.csv"

diagnos_org = pd.read_csv(root_dir+data_name)

# Create an auxilear variable
diagnosis = np.zeros(diagnos_org.shape[0])

# get the diagnosis for the different columns.
diagnosis = np.nan_to_num(np.array(diagnos_org["DXCHANGE"])) + np.nan_to_num(np.array(diagnos_org["DXCURREN"]))+ np.nan_to_num(np.array(diagnos_org["DIAGNOSIS"]))

# Change the Current diagnosis for unify the diagnosis
diagnos_org["DXCURREN"] = diagnosis

# Get only the sample with the oldest date
diagnos_final = get_uniques_samples(diagnos_org)

# Get the healty participants
RID_hl = diagnos_final[diagnos_final["DXCURREN"]==1]

# Get the participants RID
RID_hl = np.array(RID_hl["RID"])

del diagnos_final , diagnos_org, diagnosis

## Load metadata
# Read file
root_dir = "MR_Image_Analysis_ADNI_1_2_GO_FS51/"

data_name = "UCSFFSX51_11_08_19.csv"

data_org = pd.read_csv(root_dir+data_name)

## Load data file
# Read file
root_dir = ""

data_name = "PTDEMOG.csv"

meta_data_org = pd.read_csv(root_dir+data_name)

del root_dir, data_name

############################################### Filtering
# Filter the data to get the healty participants
data_org = data_org[data_org["RID"].isin(RID_hl)].reset_index()

# Filter the metadata to get the healty participants
meta_data_org = meta_data_org[meta_data_org["RID"].isin(RID_hl)].reset_index()

# Get only the screening session
meta_data = meta_data_org.loc[meta_data_org["VISCODE2"].isin(["sc"])]

# Get only the sample with the oldest date
meta_data_final = get_uniques_samples(meta_data)

# Get the Roster Identity number
RID = np.array(meta_data_final["RID"])

# Keep only the data whith have matadeta
data_org = data_org[data_org["RID"].isin(RID)]

# Get only the Non-Accelerated typt of images
data_org = data_org.loc[data_org["IMAGETYPE"]=="Non-Accelerated T1"]

# Get only the sample with the oldest date
data_final = get_uniques_samples(data_org)

# Get only those patiens with data
RID = np.array(data_final["RID"])
meta_data_final = meta_data_final[meta_data_final["RID"].isin(RID)].reset_index()

## Extract features
keys = data_final.keys()
# Check get only the Thinkness Average data
TA_mask = np.zeros(keys.shape)
for n_t,_ in enumerate(keys):
    if  keys[n_t].endswith(('TA')):
        TA_mask[n_t] = 1 

TA_mask = np.bool8(TA_mask)

# Extract TA data
data_data = data_final.loc[:,TA_mask]

#### Get the metadate
Site = meta_data_final["SITEID"]
Gender = meta_data_final["PTGENDER"]

# Calculate the Age of the participants
Age = np.zeros(meta_data_final["USERDATE"].shape)
for a ,_ in enumerate(meta_data_final["USERDATE"]):
    Age[a] = int(np.array(meta_data_final["USERDATE"][a][0:4])) - meta_data_final["PTDOBYY"][a]


# a Few samples have age 0
Age_mask = np.bool8(Age)

data = np.array(data_data[Age_mask])
Age = Age[Age_mask]
Gender = np.array(Gender[Age_mask])
Site = np.array(Site[Age_mask])

# remove intermediate variables
del a, data_final , Age_mask, keys , data_org, meta_data_final
del n_t, RID, RID_hl, TA_mask, meta_data, meta_data_org, data_data

# Make sure the data and the metadata have the same shape
assert Age.shape[0] == Gender.shape[0] == Site.shape[0] == data.shape[0]

print("Final number of participants: " + str(data.shape[0]))
print("Final number of features: " + str(data.shape[1]))
 
