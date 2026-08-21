import csv

def save_info_executions(file_name, agent_file, i, directory, *args):
	with open(file_name, 'a') as object:
		writer_object = csv.writer(object)

		data = [agent_file, i, directory]
		data.extend(list(args))

		writer_object.writerow(data)

		object.close()