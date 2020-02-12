import pickle
import uuid
import datetime
import glob

RESULT_DIR = "results/genmatch/"

def persist_result(name, data):
    timestamp = str(datetime.datetime.now())[:-7]
    file_name = name + "-" + timestamp + ".pkl"
    file_path = RESULT_DIR + file_name

    with open(file_path, "wb") as res_file:
        pickle.dump(data, res_file)

    return file_name


def read_result(base_name):
    if base_name[-3:] == "pkl":
        file_path = RESULT_DIR + base_name
    else:
        file_path_pattern = RESULT_DIR + f"{base_name}*.pkl"
        file_paths = sorted(glob.glob(file_path_pattern))
        if len(file_paths) == 0:
            raise Exception(f"No results for name: {base_name}")

        file_path = file_paths[-1] # most recent

    print("Fetching result from path:", file_path)
    with open(file_path, "rb") as res_file:
        result_entry = pickle.load(res_file)

    return result_entry

# NOTE: this code can be used to mock the Multiprocessing Manager
# objects to read pickled proxy objects.

# import sys
# mm = list(filter(lambda x: x.startswith("multi"), sys.modules))
# print(mm)
# class A():
#     pass
# #     NEWOBJ = object()

#     class RebuildProxy():
#         def __init__(self, *args, **kwargs):
#             print(args, kwargs)

#         def __getstate__(self):
#             return {}

#         def __setstate__(self, state):
#             print(state)

#     class ListProxy():
#         def __init__(self, *args, **kwargs):
#             print(args, kwargs)

#         def __getstate__(self):
#             return {}

#         def __setstate__(self, state):
#             print(state)

#     class Token():
#         def __init__(self, *args, **kwargs):
#             print(args, kwargs)

#         def __getstate__(self):
#             return {}

#         def __setstate__(self, state):
#             print(state)


# sys.modules['multiprocessing.managers'] = A
# # for m in mm:
# #     print(m)
# #     del sys.modules[m]
