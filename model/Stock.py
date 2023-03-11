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
        return f"Name: {self.id}, Date: {self.date}, Closed: {self.close}, Percentage: {self.percentage}" \
               f", Open: {self.open}, High: {self.high}, Low: {self.low}, Mass: {self.mass}, Turnover: {self.turnover}"


    def print(self):
        return self.close + ", " + self.percentage + ", "+ self.date





