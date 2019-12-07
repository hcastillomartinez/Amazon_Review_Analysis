import random
import os
import csv

p = .35
directory = r'C:\Users\damia\PycharmProjects\Amazon_Review_Analysis\data'
# directory = r'C:\Users\damia\Desktop\new_samples'
out = "C:\\Users\\damia\\PycharmProjects\\Amazon_Review_Analysis\\data\\"
# out = "C:\\Users\\damia\\Desktop\\new_samples\\"
out2 = "C:\\Users\\damia\\PycharmProjects\\Amazon_Review_Analysis\\mod_data\\"
header = 1
sample = 1
seen = []
for x in range(sample):
    tsvfileout = open(out2 + 'no_dup_sample_' + str(x) + '.tsv', 'w', encoding='Latin-1')
    writer = csv.writer(tsvfileout, delimiter='\t', lineterminator='\n')
    for filename in os.listdir(directory):
        header = 1
        print(filename)
        print("Current sample %d" % x)
        tsvfile = open(out + filename, 'r', encoding='Latin-1')
        reader = csv.reader(tsvfile, delimiter='\t', quoting=csv.QUOTE_NONE)
        for row in reader:
            if header == 1:
                writer.writerow(row)
                header = 0
            else:
                if (random.random()) > p:
                    writer.writerow(row)

