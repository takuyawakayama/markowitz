print("Importing packages...")
from pprint import pprint
import pandas as pd
import numpy as np
import urllib2
import json
import csv
from alpha_vantage.timeseries import TimeSeries
from pandas.io.json import json_normalize
import cvxopt as opt
from cvxopt import blas, solvers

solvers.options['show_progress'] = False

print("Import Packages complete.")

#Set the total amount we have to invest 
total_invst = 100000.0 #Not really important. 100,000 USD
#leverage = 2.5 #Leverage you can make on portfolio (short)

apikey = 'AU6S38KFU04HJB8S'
outputsize = 'full'

#Set initial conditions:
histdays = 252*1 #How many days in history to look
futuredays =252 #Length of buy-hold strategy. 252 trading days in 1 year

print "Initial data loading complete. Running programs"


################################################
def main():
    """Main program 
    """
    
    #Import historical stock data
    totaldatadf = pd.read_csv('/Users/takuyawakayama/Desktop/Columbia/APMA4903 Seminar/totaldata.csv')
            
    #Converting Dates into datetime format
    totaldatadf['Date']=pd.to_datetime(totaldatadf['Date'])
    datecolumn = totaldatadf['Date'].iloc[::-1]
    datecolumn = datecolumn.reset_index(drop=True)
    totaldatadf.index = totaldatadf['Date']
    totaldatadf.drop('Date', axis=1, inplace=True)
    
    #Reorder..
    totaldatadf =totaldatadf.iloc[::-1]
    print "Dataframe complete." ##Now, oldest date on top, newest date on bottom
        
    #Compute Monthly %changes. 0.05=5%
    monthlypctchg = totaldatadf.pct_change(periods=21)[21:]
    totalrows = monthlypctchg.shape[0]    
    
    #Resize stock prices to monthlypctchg matrix
    stockpricedf = totaldatadf[21:]
    
    print 'Done with initial matrices'
    
    #Import the list of DJIA companies during the 17year span
    #This creates a matrix of dates on top, and then 30 companies in DJIA
    djiacompanies=pd.read_csv('/Users/takuyawakayama/Desktop/Columbia/APMA4903 Seminar/djia-components.csv')

    #Create list of dates that there was an index change
    #type=DatetimeIndex, list of dates of form datetime64[ns]
    switchdates = pd.to_datetime(list(djiacompanies))
    
    #Initialize dictionary of ROIs
    profitdict = {}
    
    #For each day we backtest...
    for i in range(21+histdays,totalrows-futuredays):
        currentdate = datecolumn[i]
        print currentdate
        #Find the appropriate list of 30 companies in DJIA on start day
        for searchdate in switchdates:
            if currentdate > searchdate:
                stocklistdf = djiacompanies[searchdate.strftime('%-m/%-d/%Y')]
                break
        if not 'stocklist' in locals():
            stocklistdf = djiacompanies['1/27/2003']
        
        #Creates list of the 30 stocks
        stocklist = stocklistdf.values.flatten().tolist()
        
        #Pulls the %change data for the relevant dates
        returndata = monthlypctchg[stocklist].iloc[i-histdays-21+1:i-21+1,:]
        returndata = returndata.dropna(axis=1, how='any')
        
        #Create the pairwise covariance matrix
        covmatrix = returndata.cov()
        
        #Now, we find optimum weightings
        
        #Format to pass onto CVXOPT
        n_assets = covmatrix.shape[0]
        n_obs = histdays
        returns = np.asmatrix(returndata)
        N=100
        mus = [10**(5.0*t/N -1.0) for t in range(80,N)]
        
        S=opt.matrix(covmatrix.as_matrix())
        pbar = opt.matrix(np.mean(returndata,axis=0))
        
        """WITHOUT SHORT SELLING"""
        G = -opt.matrix(np.eye(n_assets)) #Convert constraint to form Cx<=D because we want min bound
        h = opt.matrix(0.0, (n_assets,1)) #Restricts lower bound to 0 to prevent short selling
        A = opt.matrix(1.0, (1,n_assets))
        b = opt.matrix(1.0)               #Total weight of portfolio = 1.0

        """WITH SHORTSELLING"""
        #G = 0.0*opt.matrix(np.vstack((-np.eye(n_assets),np.eye(n_assets)))) #Convert constraint to form Cx<=D because we want min bound
        #h = opt.matrix(0.0, (2*n_assets,1)) #Restricts lower bound to 0 to prevent short selling
        #A = opt.matrix(1.0, (1,n_assets))
        #b = opt.matrix(1.0)               #Total weight of portfolio = 1.0        

        #Optimize onto CVXOPT
        ##Create portfolios
        ###Note: Gx<=h, Ax=b are constraints
        portfolios = [solvers.qp(mu*S, -pbar, G,h, A, b)['x'] for mu in mus]        
        returns = [blas.dot(pbar, x) for x in portfolios]
        risks = [np.sqrt(blas.dot(x, S*x)) for x in portfolios]
        
        ## Calculate the 2nd Degree polynomial of the frontier curve
        m1 = np.polyfit(returns, risks, 2)
        x1 = np.sqrt(m1[2] / m1[0])
        
        ## Calculate optimal portfolio
        wt = solvers.qp(opt.matrix(x1 * S), -pbar, G, h, A, b)['x'] 
        weights = np.asarray(wt)
        weights = weights.T

        stddevs = returndata.std()
        meanrets = returndata.mean()
        stocklist = meanrets.index.tolist()
        
        #Compute investment amount in each stock
        invstamt = pd.DataFrame(weights*total_invst)
        invstamt.columns = stocklist
        #gross_invst = float(np.sum(np.absolute(invstamt),axis=1))
        #invstamt = invstamt/gross_invst*total_invst*leverage
        
        #Obtain current stockprice, as well as stock price at end of investment
        currstkpr = stockpricedf.iloc[[i-21]]
        resultstkpr = stockpricedf.iloc[[i+futuredays-21]]
        currstkpr = currstkpr[stocklist]
        resultstkpr = resultstkpr[stocklist]
        
        #Adjust for missing data, as well as delisting from NYSE
        nancols = pd.isnull(resultstkpr).sum()>0        
        stocks_with_nan = nancols.index[nancols==True].tolist()
        if len(stocks_with_nan)>0:
            for stock in stocks_with_nan:
                lastpriceindex = stockpricedf[stock].iloc[i-21:i+futuredays-21].last_valid_index()
                lastprice = float(stockpricedf[stock].loc[[lastpriceindex]])
                resultstkpr[stock] = lastprice
        
        #Rate of change. 1 would mean there is no change.
        ratechange = resultstkpr.reset_index(drop=True)/currstkpr.reset_index(drop=True)
        
        #Value of portfolio at end of investment
        endvalue = ratechange*invstamt
        
        #Create result matrix that contains all information we might need
        resultmatrixxx = pd.concat([currstkpr.T, resultstkpr.T, ratechange.T, invstamt.T, endvalue.T], axis=1)
        resultmatrixxx.columns=['current', 'result', 'rate', 'invst', 'endvalue']
        
        ##Print to Python shell to monitor progress
        print '------------------------------------'
        print i
        print currentdate      
        print resultmatrixxx
        
        
        #Compute return on investment. 0 means no change, positive is increase
        ROI = float((endvalue.sum(axis=1)-total_invst)/total_invst)
        
        #Record data for this run
        profitdict[currentdate] = ROI
        print ROI
        
    
    
    #Ensure ordering of series is in order of date    
    ROIseries = pd.Series(profitdict)
    ROIseries.sort_index(inplace=True)
    
    #Export to CSV
    ROIseries.to_csv("ROIresults1ynew.csv")
    
    return

if __name__ == '__main__':
    main()