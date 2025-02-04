from multiprocessing import Process

from predictionEngine.Predictor import predict
from utils.StockTicketType import StockTicketType;

if __name__ == "__main__":
    # stocks = ['TPEIR.AT','ETE.AT','EUROB.AT','ALPHA.AT',
    #           'MYTIL.AT','AEGN.AT']

    stocks_sp = ['NVDA','MSFT','AAPL','AMD']


    for stock in stocks_sp:
        p = Process(target=predict, args=(stock,StockTicketType.CLOSE.value,60, 10))
        p.start()
        p.join()  # this blocks until the process terminates
        result = p.exitcode
        print(f"********************* Returned status: {result} *********************")
