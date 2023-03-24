from multiprocessing import Process

from predictionEngine.Predictor import predict

if __name__ == "__main__":
    stocks = ['MYTIL.AT','TPEIR.AT','NBGIF','GEKTERNA.AT',
              'TENERGY.AT','LAMDA.AT','PPC.AT','OPAP.AT','AEGN.AT','BELA.AT']
    # PPC.AT = DEH
    # BELA.AT = Jumbo

    for stock in stocks:
        p = Process(target=predict, args=(stock,'Close',60, 50))
        p.start()
        p.join()  # this blocks until the process terminates
        result = p.exitcode
        print(f"********************* Returned status: {result} *********************")
