#!/usr/bin/env python3
#
#
import os
import pickle
from datetime import datetime

timestamp = datetime.now().strftime("%Y.%m.%d_%H-%M-%S")

OLD_PICKLE = "data_sorted.pickle"
NEW_PICKLE = "data_sorted-retrain.pickle"

timestamp = datetime.now().strftime("%Y.%m.%d_%H-%M-%S")

MERGED_PICKLE = f"training-data-{timestamp}.pickle"

if os.path.isfile(OLD_PICKLE):
    print(f"Opening {OLD_PICKLE}")
    with open(OLD_PICKLE, 'rb') as oldf:
         data_old = pickle.load(oldf)
else:
    print(f"File {OLD_PICKLE} not found")
    sys.exit()

print(f"Old dict size {len(data_old)}")

if os.path.isfile(NEW_PICKLE):
    print(f"Opening {NEW_PICKLE}")
    with open(NEW_PICKLE, 'rb') as newf:
         data_new = pickle.load(newf)
else:
    print(f"File {OLD_PICKLE} not found")
    sys.exit()

print(f"New dict size {len(data_new)}")


data_old.update(data_new)
print(f"Merged dict size {len(data_old)}")

with open(MERGED_PICKLE, "wb") as save:
    pickle.dump(data_old, save, pickle.HIGHEST_PROTOCOL)
