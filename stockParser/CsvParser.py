# Read CSV file
import csv

from model.Stock import Stock
from datetime import datetime,timedelta

stocks = []

def parseAllYears(id):
    with open('../resources/'+id+'.csv', encoding="UTF-8") as fp:
        reader = csv.reader(fp, delimiter=",", quotechar='"')
        # next(reader, None)  # skip the headers
        next(reader)
        next(reader)
        for row in reader:
            stock = Stock(id, row[0],row[1], row[2], row[3], row[4], row[5], row[6], row[7])
            stocks.append(stock)
        return stocks


def parseSpecificYears(id,days):
    today = datetime.today()
    with open('../resources/'+id+'.csv', encoding="UTF-8") as fp:
        reader = csv.reader(fp, delimiter=",", quotechar='"')
        # next(reader, None)  # skip the headers
        next(reader)
        next(reader)
        for row in reader:
            date = datetime.strptime(row[0],'%d/%m/%Y')

            if(date >= today -timedelta(days=days)):
                stock = Stock(id, row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7])
                stocks.append(stock)
        return stocks

def replaceGreekCsvHeaders(id):
    with open('../resources/'+id+'.csv', encoding="UTF-8") as inFile, \
            open('../resources/'+id+'_eng.csv','w+',newline='',encoding='UTF-8') as outfile:
        r = csv.reader(inFile)
        w = csv.writer(outfile)

        next(r, None)  # skip the first row from the reader, the old header
        next(r, None)

        w.writerow([id])
        w.writerow(['date', 'close', 'percentage', 'open', 'high','low','volume','turnover'])

        # copy the rest
        for row in r:
            w.writerow(row)


if __name__ == "__main__":
    replaceGreekCsvHeaders('mytil')

