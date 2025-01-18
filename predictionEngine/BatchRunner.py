from multiprocessing import Process

from predictionEngine.Predictor import predict
from utils.StockTicketType import StockTicketType;

if __name__ == "__main__":
    stocks = ['TPEIR.AT','ETE.AT','EUROB.AT','ALPHA.AT',
              'MYTIL.AT','AEGN.AT']

    stocks_short = ['ALPHA.AT','TENERGY.AT','LAMDA.AT']
    # PPC.AT = DEH
    # BELA.AT = Jumbo

    for stock in stocks:
        p = Process(target=predict, args=(stock,StockTicketType.HIGH.value,90, 50))
        p.start()
        p.join()  # this blocks until the process terminates
        result = p.exitcode
        print(f"********************* Returned status: {result} *********************")
