from textblob import TextBlob
import csv
import sys

csv.field_size_limit(100000000)
with open(sys.argv[1],'r',encoding='Latin-1') as tsvfile, open('modified_'+sys.argv[1],'w',encoding='Latin-1') as tsvfileout:
    reader = csv.reader(tsvfile, delimiter='\t',quoting=csv.QUOTE_NONE)
    writer = csv.writer(tsvfileout, delimiter='\t', lineterminator='\n')
    row_head = 1
    for row in reader:
        #Writing the headers
        if row_head == 1:
            row.append('sentiment_rating')
            row.append('subjectivity_rating')
            row_head = 0
            writer.writerow(row)
        else:
            #Writing the other columns
            review = TextBlob(row[13])
            review.sentiment
            row.append(review.sentiment.polarity)
            row.append(review.sentiment.subjectivity)
            writer.writerow(row)
