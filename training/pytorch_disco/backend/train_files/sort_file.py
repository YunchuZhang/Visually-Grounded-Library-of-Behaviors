# parse file

file_names = []
with open("train_file_controller.txt", 'r') as f:
	for line in f:
		file_names.append(line.strip().replace('\n', ''))


with open("val_file_controller.txt", 'r') as f:
	for line in f:
		file_names.append(line.strip().replace('\n', ''))



file_names.sort()
for file in file_names:
	print(file)

print("hello")
