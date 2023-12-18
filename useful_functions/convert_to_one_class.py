import os

input_dir_path = '/home/eiman/data/resized/train/labels/'
output_dir_path = '/home/eiman/data/resized/train_s/labels/'
isExist = os.path.exists(output_dir_path)
if not isExist:
   # Create a new directory because it does not exist
   os.makedirs(output_dir_path)

list_dir = os.listdir(input_dir_path)

for filename in list_dir:

    replaced_content = "" # An empty write file
    if filename[-4:] == '.txt': # So that no other file than text files are read
        with open(input_dir_path + filename, 'r') as read_file:
            for line in read_file:
                line = line.strip()
                replaced_content = replaced_content + '0' + line[1::] + "\n"
        read_file.close()

        with open(output_dir_path + filename, 'w') as write_file:
            write_file.write(replaced_content)
        write_file.close()
