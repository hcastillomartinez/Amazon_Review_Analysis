import random
import os
import csv
import numpy

directory = r'C:\Users\damia\PycharmProjects\Amazon_Review_Analysis\data'
# directory = r'C:\Users\damia\Desktop\pp'
out = "C:\\Users\\damia\\PycharmProjects\\Amazon_Review_Analysis\\data\\"
# out = "C:\\Users\\damia\\Desktop\\pp\\"
out2 = "C:\\Users\\damia\\PycharmProjects\\Amazon_Review_Analysis\\mod_data\\"
header = 1
sample = 1
tot_row = 18717800
first = 1
# row[8] is helpful votes
# 18717800 number of rows

def row_count(filename):
    row_c = 0
    tsvfile1 = open(out + filename, 'r', encoding='Latin-1')
    reader1 = csv.reader(tsvfile1, delimiter='\t', quoting=csv.QUOTE_NONE)
    for row1 in reader1:
        row_c += 1
    return row_c-1


for x in range(sample):
    tsvfileout = open(out2 + 'sample_' + str(1) + '.tsv', 'w', encoding='Latin-1')
    writer = csv.writer(tsvfileout, delimiter='\t', lineterminator='\n')
    for filename in os.listdir(directory):
        print(filename)
        curr_row_count = row_count(filename)
        header = 1
        tsvfile = open(out + filename, 'r', encoding='Latin-1')
        reader = csv.reader(tsvfile, delimiter='\t', quoting=csv.QUOTE_NONE)

        for row in reader:
            if header == 1:
                if first == 1:
                    writer.writerow(row)
                header = 0
                first = 0
            else:
                # p = numpy.tanh(int(row[8])*(curr_row_count/tot_row)+int(row[8]))
                if int(row[8]) == 0:
                    p = numpy.tanh(numpy.log(1)+.15)-curr_row_count/tot_row
                else:
                    p = numpy.tanh(numpy.log(int(row[8])))-curr_row_count/tot_row
                if (random.random()) <= p:
                    writer.writerow(row)

# print("avg number of duplicates blocked = %d" % hit/len(seen))
