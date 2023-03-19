from multiprocessing import Process

from predictionEngine.Predictor import predict

if __name__ == "__main__":
    stocks = ['DEI','LAMDA.AT','MYTIL.AT','BELA.AT','GEKTERNA.AT','AEGN.AT','TPEIR.AT','TENERGY.AT','NBGIF']

    for stock in stocks:
        p = Process(target=predict, args=(stock,'Close',60, 50))
        p.start()
        p.join()  # this blocks until the process terminates
        result = p.exitcode
        print(f"********************* Returned status: {result} *********************")
