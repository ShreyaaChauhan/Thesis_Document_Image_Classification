import os

rootdir = (
    "/Users/shreyachauhan/Thesis_Document_Image_Classification/model/data/Tobacco3482"
)
sub_dir = os.listdir(rootdir)  # list of subdirectories and files
sub_dirs_path = [os.path.join(rootdir, x) for x in sub_dir]
print(sub_dirs)
