from enum import Enum

class StockTicketType(Enum):
    OPEN = "Open"
    CLOSE = "Close"
    HIGH = "High"
    LOW = "Low"
    VOLUME = "Volume"