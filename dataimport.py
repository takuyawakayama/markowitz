###dataimport.py
print("Importing packages...")
from pprint import pprint
import pandas as pd
import numpy as np
import urllib2
import json
import csv
import time
from alpha_vantage.timeseries import TimeSeries
from pandas.io.json import json_normalize

print("Import Packages complete.")

def dataimport():
    ###Main program of this python file that imports data from alphavantage,
    ###and exports a csv file with historical data of the listed stocks
    
    #Import the list of DJIA companies during the 17year span
    #This creates a matrix of dates on top, and then 30 companies in DJIA
    djiacompanies=pd.read_csv('/Users/takuyawakayama/Desktop/Columbia/APMA4903 Seminar/djia-components.csv')

    #Create list of dates that there was an index change
    #type=DatetimeIndex, list of dates of form datetime64[ns]
    switchdates = pd.to_datetime(list(djiacompanies))
    
    #Lists all companies that we might ever use. type = LIST
    allcompanies = pd.Series(djiacompanies.values.ravel()).unique().tolist()
    
    #API key for Alphavantage
    apikey = 'AU6S38KFU04HJB8S'
    outputsize = 'full'
    
    #Initialize dataframe using AAPL as sample company
    ts = TimeSeries(key=apikey, output_format='pandas')
    totaldatadf, meta_data = ts.get_daily_adjusted(symbol='AAPL', outputsize=outputsize) 
    columns = ['low', 'open', 'high', 'close', 'volume', 'split coefficient', 'dividend amount']
    totaldatadf.drop(columns, inplace=True, axis=1)   
    totaldatadf=totaldatadf.rename(columns = {'adjusted close':'nan'})
    
    #Create dataframe of historical daily prices for each company ever in DJIA
    for company in allcompanies:     
        
        #Sometimes gives Error 503, so force program to complete
        connected = False
        while not connected:
            try:
                companydatadf, meta_data = ts.get_daily_adjusted(symbol=company, outputsize=outputsize) 
                columns = ['low', 'open', 'high', 'close', 'volume', 'split coefficient', 'dividend amount']
                companydatadf.drop(columns, inplace=True, axis=1)   
                companydatadf=companydatadf.rename(columns = {'adjusted close':company})
                totaldatadf = pd.concat([totaldatadf,companydatadf],axis=1)
                connected = True
            except:
                time.sleep(10)
                pass
    
    #Delete sample column
    totaldatadf.drop('nan', axis=1, inplace=True)
    
    #Reorder
    totaldatadf=totaldatadf.iloc[::-1]
    
    #Export to CSV
    totaldatadf.to_csv("totaldata.csv")
    return

if __name__ == '__main__':
    dataimport()