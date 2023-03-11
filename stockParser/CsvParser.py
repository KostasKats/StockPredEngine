# Read CSV file
import csv

from model.Stock import Stock
from datetime import datetime,timedelta

stocks = []

def parseAllYears(id):
    with open('../resources/mytil/HistoryCloses'+id+'all.csv', encoding="UTF-8") as fp:
        reader = csv.reader(fp, delimiter=",", quotechar='"')
        # next(reader, None)  # skip the headers
        next(reader)
        next(reader)
        for row in reader:
            stock = Stock("MYTIL", row[0],row[1], row[2], row[3], row[4], row[5], row[6], row[7])
            stocks.append(stock)
        return stocks


def parseSpecificYears(id,days):
    today = datetime.today()
    with open('../resources/mytil/HistoryCloses'+id+'all.csv', encoding="UTF-8") as fp:
        reader = csv.reader(fp, delimiter=",", quotechar='"')
        # next(reader, None)  # skip the headers
        next(reader)
        next(reader)
        for row in reader:
            date = datetime.strptime(row[0],'%d/%m/%Y')

            if(date >= today -timedelta(days=days)):
                stock = Stock("MYTIL", row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7])
                stocks.append(stock)
        return stocks

if __name__ == "__main__":
    for stock in parseSpecificYears('ΜΥΤΙΛ',365):
        print(stock.__str__())

