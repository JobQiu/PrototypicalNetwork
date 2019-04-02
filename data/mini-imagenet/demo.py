import numpy as np

np.random.seed(42)
from bs4 import BeautifulSoup

with open('empty_dir') as f:
    content = f.readlines()
# you may also want to remove whitespace characters like `\n` at the end of each line
content = [x.strip() for x in content]
content = [x.split('/')[-1] for x in content]
content = [x.split('.')[0] for x in content]
content = set(content)

train_list = [
    "n04509417", "n07697537", "n03924679", "n02113712", "n02089867", "n02108551", "n03888605", "n04435653", "n02074367",
    "n03017168", "n02823428", "n03337140", "n02687172", "n02108089", "n13133613", "n01704323", "n02165456", "n01532829",
    "n02108915", "n01910747", "n03220513", "n03047690", "n02457408", "n01843383", "n02747177", "n01749939", "n13054560",
    "n03207743", "n02105505", "n02101006", "n09246464", "n04251144", "n04389033", "n03347037", "n04067472", "n01558993",
    "n07584110", "n03062245", "n04515003", "n03998194", "n04258138", "n06794110", "n02966193", "n02795169", "n03476684",
    "n03400231", "n01770081", "n02120079", "n03527444", "n07747607", "n04443257", "n04275548", "n04596742", "n02606052",
    "n04612504", "n04604644", "n04243546", "n03676483", "n03854065", "n02111277", "n04296562", "n03908618", "n02091831",
    "n03838899"]

test_list = [
    "n02110341", "n01930112", "n02219486", "n02443484", "n01981276", "n02129165", "n04522168", "n02099601", "n03775546",
    "n02110063", "n02116738", "n03146219", "n02871525", "n03127925", "n03544143", "n03272010", "n07613480", "n04146614",
    "n04418357", "n04149813"]

train_list = [x for x in train_list if x not in content]
test_list = [x for x in test_list if x not in content]

exist = [x for x in content] + [x for x in train_list] + [x for x in test_list]
exist = set(exist)

print(" ".join(train_list))
print(" ".join(test_list))
print(len(train_list))
print(len(test_list))

with open('ReleaseStatus.xml') as infile:
    soup = BeautifulSoup(infile, 'lxml')

synsets = soup.find_all('synset')

synsets = [synsets[i].attrs['wnid'] for i in range(len(synsets)) if int(synsets[i].attrs['numimages']) > 100]

extra_train = np.random.choice(synsets, 100)
extra_test = np.random.choice(synsets, 50)
extra_train2 = np.random.choice(synsets, 100)
extra_train3 = np.random.choice(synsets, 100)
extra_train4 = np.random.choice(synsets, 100)

extra_train = [x for x in extra_train if x not in exist]
extra_test = [x for x in extra_test if x not in exist and x not in extra_train]
extra_train2 = [x for x in extra_train2 if x not in exist and x not in extra_train and x not in extra_test]
extra_train3 = [x for x in extra_train3 if
                x not in exist and x not in extra_train and x not in extra_test and x not in extra_train2]
extra_train4 = [x for x in extra_train4 if
                x not in exist and x not in extra_train and x not in extra_test and x not in extra_train2 and x not in extra_train3]

print("\" \"".join(train_list + extra_train))
print(" ".join(test_list + extra_test))

print(" ".join(extra_train))

extra_train_str = ""
for e in extra_train4:
    extra_train_str += "\"" + e + "\"" + " "
print(extra_train_str)
"""

extra_test_str = ""
for e in extra_test:
    extra_test_str += "\"" + e + "\"" + " "
print(extra_test_str)

# generate train and test list/
with open("train_list", 'w') as fp:
    for train in train_list + extra_train:
        fp.write(train)

with open("test_list", 'w') as fp:
    for train in test_list + extra_test:
        fp.write(train)
        


"""
