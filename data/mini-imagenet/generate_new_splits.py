import os

type = 'test'

dirs = next(os.walk('data/' + type))[1]

empty_dirs = [x for x in dirs if not os.listdir('data/' + type + "/" + x)]

print(empty_dirs)
print(len(empty_dirs) / len(dirs))

print(len(dirs) - len(empty_dirs))
