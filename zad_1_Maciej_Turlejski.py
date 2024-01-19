import pandas as pd
data = pd.read_csv("TravelInsurancePrediction.csv")
average_age = data["Age"].mean()
print("The arithmetic average of the age of customers is", average_age)
