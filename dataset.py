import kagglehub
import shutil

default_path = kagglehub.dataset_download("jacksoncrow/stock-market-dataset")

my_path = "/home/great/Documents/share price predection/"

shutil.move(default_path, my_path)

print(f"Moved dataset to {my_path}")
