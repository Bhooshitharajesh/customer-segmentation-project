from pydantic import BaseModel 

class InputData(BaseModel):
     Customer_ID: float
     Invoice: str
     Quantity: int
     Price: float
     InvoiceDate: str
     StockCode: str
     Country: str
     Description: str
     
     
   