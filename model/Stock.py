class Stock:
    def __init__(self,id,date,close,percentage,open,high,low,mass,turnover):
        self.id = id
        self.date = date
        self.close = close
        self.percentage = percentage
        self.open = open
        self.high = high
        self.low = low
        self.mass = mass
        self.turnover = turnover

    def __str__(self):
        return self.close + ", " + self.percentage + ", "+ self.date






