import kagglehub

# Download latest version
path = kagglehub.dataset_download("kneroma/tacotrashdataset")

print("Path to dataset files:", path)
