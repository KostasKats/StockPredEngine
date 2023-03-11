import stockParser
from stockParser.CsvParser import parseAllYears,parseSpecificYears

if __name__ == "__main__":
    for stock in parseSpecificYears(4):
        print(stock.__str__())