import random
import os
import csv

p = 0.35
directory = r'C:\Users\damia\PycharmProjects\Amazon_Review_Analysis\data'
out = "C:\\Users\\damia\\PycharmProjects\\Amazon_Review_Analysis\\data\\"
out2 = "C:\\Users\\damia\\PycharmProjects\\Amazon_Review_Analysis\\mod_data\\"
header = 1
sample = 8
for x in range(sample):
    for filename in os.listdir(directory):
        header = 1
        with open(out + filename, 'r', encoding='Latin-1') as tsvfile, open(out2 + 'sample_'+str(x)+'.tsv', 'w',
                                                                            encoding='Latin-1') as tsvfileout:
            reader = csv.reader(tsvfile, delimiter='\t', quoting=csv.QUOTE_NONE)
            writer = csv.writer(tsvfileout, delimiter='\t', lineterminator='\n')
            for row in reader:
                if header == 1:
                    writer.writerow(row)
                    header = 0
                else:
                    if (random.random()) > p:
                        writer.writerow(row)

