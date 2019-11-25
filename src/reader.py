from textblob import TextBlob
import csv
import os


import sys
#C:\Users\damia\PycharmProjects\Amazon_Review_Analysis\src
csv.field_size_limit(100000000)
directory = r'C:\Users\damia\PycharmProjects\Amazon_Review_Analysis\data'
row_max = 500000
for filename in os.listdir(directory):
    out = "C:\\Users\\damia\\PycharmProjects\\Amazon_Review_Analysis\\data\\"
    out2 = "C:\\Users\\damia\\PycharmProjects\\Amazon_Review_Analysis\\mod_data\\"
    out3 = "D:\\"
    with open(out+filename,'r',encoding='Latin-1') as tsvfile, open(out2+'new_modified_'+filename,'w',encoding='Latin-1') as tsvfileout:
        curr_row = 0
        reader = csv.reader(tsvfile, delimiter='\t',quoting=csv.QUOTE_NONE)
        writer = csv.writer(tsvfileout, delimiter='\t', lineterminator='\n')
        row_head = 1

        for row in reader:
            #Writing the headers
            if row_head == 1:
                row.append('word_count')
                row.append('sentence_count')
                row.append('adjective_count')
                row.append('adverb_count')
                row.append('noun_count')
                row.append('pronoun_count')
                row.append('verb_count')
                row_head = 0
                writer.writerow(row)
            else:
                #Writing the other columns
                review = TextBlob(row[13])
                tags = review.tags
                sentences = review.sentences
                adjectives = 0
                adverbs = 0
                nouns = 0
                pronouns = 0
                verb = 0
                sentence_count = len(review.sentences)
                word_count = len(review.words)
                for n in tags:
                    if n[1] == "JJ" or n[1] == "JJR" or n[1] == "JJS":
                        adjectives += 1
                    if n[1] == "RB" or n[1] == "RBR" or n[1] == "RBS" or n[1] == "RP":
                        adverbs += 1
                    if n[1] == "PRP" or n[1] == "PRP$":
                        nouns += 1
                    if n[1] == "JJ" or n[1] == "JJR" or n[1] == "JJS":
                        pronouns += 1
                    if n[1] == "MD" or n[1] == "VB" or n[1] == "VBZ" or n[1] == "VBP" or n[1] == "VBD" or n[
                        1] == "VBN" or n[1] == "VBG":
                        verb += 1
                row.append(word_count)
                row.append(sentence_count)
                row.append(adjectives)
                row.append(adverbs)
                row.append(nouns)
                row.append(pronouns)
                row.append(verb)
                curr_row += 1
                writer.writerow(row)
                if curr_row >= row_max:
                    break
