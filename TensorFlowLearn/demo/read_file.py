import os


def list_filename(path):
    file_path_list = []
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        # print(file_path)
        # if file.startswith('._'):
        #     pass
        # else:
        #     file_path_list.append(file_path)
        if not file.startswith('._'):
                file_path_list.append(file_path)
    return file_path_list


test_path = r'E:\PrivateDocuments\object_recognition_data\test_data'
path_list = list_filename(test_path)
print(len(path_list))

print(os.listdir(test_path)[2000])
