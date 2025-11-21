from sklearn.datasets import fetch_lfw_people
import os

print("Downloading LFW dataset...")
# Using min_faces_per_person=1 to get more diverse faces for the 50 images.
# resize=0.4 is a default value, it's fine for our purpose.
lfw_people = fetch_lfw_people(min_faces_per_person=1, resize=0.4)
print("Download complete.")

from sklearn.datasets import get_data_home
data_home = get_data_home()
print(f"Scikit-learn data home: {data_home}")

lfw_dir = os.path.join(data_home, 'lfw_home')
print(f"LFW data should be in: {lfw_dir}")

# The structure is typically lfw_home/lfw_funneled/
# We just need the root path, the 'find' command later will get the images.
if os.path.exists(lfw_dir):
    print(f"Found LFW home directory at: {lfw_dir}")
else:
    print("LFW home directory not found where expected.")
