# -*- coding: UTF-8 -*-
import os
import pandas as pd
from datetime import datetime, date, time, timezone, timedelta
from func import *

class MarketInfo:
    def __init__(self):
        self.stockdata = dict()
    def load(self, dir = './stockdata'):
        for fn in os.listdir(dir):
            self.stockdata[fn.replace('.TW.csv','')] = pd.read_csv(os.path.join(dir,fn))
        print(self.stockdata.keys())
    def query(self, stocknum, begin_date = '2018-09-18', offset = 1):
        def binarysearch(lst,idx_lower,idx_upper, date_target, depth = 0):
            idx_cur = int( (idx_lower+idx_upper)/2 )
            date_cur = datetime.strptime(lst[idx_cur], '%Y-%m-%d')
            #print(date_cur, date_target, idx_cur,idx_lower,idx_upper, )
            if depth >= 15:
                return -1
                
            if date_cur == date_target:
                return idx_cur
            elif date_cur > date_target:
                return binarysearch(lst,idx_lower,idx_cur,date_target, depth+ 1)
            elif date_cur < date_target:
                return binarysearch(lst,idx_cur,idx_upper,date_target, depth + 1)

        target = self.stockdata[stocknum]
        #Date Open High Low Close Adj Close Volume

        idx = binarysearch(target['Date'], 0, len(target['Date']), datetime.strptime(begin_date, '%Y-%m-%d'))
        if idx == -1:
            return [],-1
        return target.iloc[idx:idx+offset], idx

#print(M.query('2330', '2018-09-18'))


class MarketAccount:
    def __init__(self, balance = 1000000):
        self.stocks = dict()
        self.balance = balance
        self.log = list()
    def buy(self, stocknum, date, price, budget = 0, volume = 0, amount_type = 'budget'):
        
        TheDay,idx = M.query(stocknum,begin_date=date)
        
        tradeprice = min( TheDay['Open'][idx] , price )
        
        amount = int(budget/tradeprice) if amount_type == 'budget' else volume
        
        if self.balance > tradeprice * amount:
            if price < TheDay['High'][idx] or price > TheDay['Low'][idx]:
                self.balance -= tradeprice * amount
                if stocknum not in self.stocks.keys():
                    self.stocks[stocknum] = 0
                self.stocks[stocknum] += amount
                self.log.append('buy, stocknum {}, date {}, price {}, amount {}, success , balance {}'.format( stocknum, date, tradeprice, amount, self.balance))                
                return True
            else:
                self.log.append('buy, stocknum {}, date {}, price {}, amount {}, trade not match'.format( stocknum, date, price, amount))
        else:
            self.log.append('buy, stocknum {}, date {}, price {}, amount {}, not enough money'.format( stocknum, date, price, amount))
        return False


    def sell(self, stocknum, date, price, amount = 0, amount_type = None,mode = None):
        if amount_type == 'all':
            amount = self.stocks[stocknum]
            if amount == 0:
                return False

        if mode == 'closeprice':
            TheDay,idx = M.query(stocknum,date)
            tradeprice = TheDay['Adj Close'][idx]
            self.balance += tradeprice * amount
            self.stocks[stocknum] -= amount
            self.log.append('sell, stocknum {}, date {}, price {}, amount {}, success, balance {}'.format( stocknum, date, tradeprice, amount, self.balance))                
            return True

        if stocknum in self.stocks.keys() and self.stocks[stocknum] >= amount:
            TheDay,idx = M.query(stocknum,date)
            if price < TheDay['High'][idx] or price > TheDay['Low'][idx]:
                tradeprice = max( TheDay['Open'][idx] , price )
                self.balance += tradeprice * amount
                self.stocks[stocknum] -= amount
                self.log.append('sell, stocknum {}, date {}, price {}, amount {}, success, balance {}'.format( stocknum, date, tradeprice, amount, self.balance))                
                return True
            else:
                self.log.append('sell, stocknum {}, date {}, price {}, amount {}, trade not match'.format( stocknum, date, price, amount))
        else:
            self.log.append('sell, stocknum {}, date {}, price {}, amount {}, not enough stock'.format( stocknum, date, price, amount))
        
        return False
    
    def showlog(self):
        for x in self.log:
            print(x)

    def sell_allstock(self,date):
        for stocknum in self.stocks.keys():
            price = 0
            self.sell(stocknum, date, price, amount = 0, amount_type = 'all')
    
    def showlog2plt(self, start_date='2016-12-31', output_figure_fname = './test.png'):
        def date_delta(a,b):
            A = datetime.strptime(a , '%Y-%m-%d')
            B = datetime.strptime(b , '%Y-%m-%d')
            #print(A-B)
            return int((A-B).days)
            
        y = []
        x = []
        plt_title = self.log[1].split(',')[1]
        for line in self.log:
            print(line)
            if 'success' in line and 'sell' in line:
                print(line.split(' ')[4])
                x.append( date_delta(line.split(' ')[4].replace(',',''),start_date) )
                y.append(float(line.split(' ')[-1]))
        
        plt.cla()
        plt.plot(x,y)
        plt.title(plt_title)
        plt.ylabel('balance')
        plt.xlabel('days')
        plt.savefig(output_figure_fname)

M = MarketInfo()
M.load('./20clean')



class TradeAgent:
    def __init__(self, name = 'robot_1'):
        self.agentname = name
        self.MA = MarketAccount(balance = 1000000)
        self.targetstocks = ['2330']
        #self.model = model

    def test_period(self, date, offset):
        pass

    def test_by_predefined_trade(self, PTfile, allin_rate = 0.25 ):
        with open(PTfile, 'r') as f:
            PT = pd.read_csv(f)
            ptdict = PT.to_dict()
            
            sss = [ x.replace('.TW','') for x in PT['Stock symbol'].to_list() ]
            bts = PT['Buying time'].to_list()
            sts = PT['Selling time'].to_list()
            print(sss)
            print(bts)
            print(sts)
            date_cur = datetime.strptime('2017-01-03', '%Y-%m-%d')
            for i in range( 2000 ) :
                today = date_cur.__str__()[:10]
                #print('today', today)
                if today == '2020-06-02':
                    self.MA.sell_allstock('2020-06-02')
                
                if today in bts:
                    for ss,bt,st in zip(sss,bts,sts):
                        if bt == today:
                            #print('bt buy', bt, ss)
                            stocknum = ss
                            price = 100000000 #反正會取min
                            budget = int(self.MA.balance * allin_rate)
                            #def buy(self, stocknum, date, price, budget = 0, volume = 0, amount_type = 'budget'):
                            self.MA.buy(stocknum, today, price, budget = budget, amount_type = 'budget')

                if today in sts:
                    #print('sell', today)
                    for ss,bt,st in zip(sss,bts,sts):
                        if st == today:
                            stocknum = ss
                            price = 0 #反正會取max
                            amount = 0 #全部賣掉
                            self.MA.sell(stocknum, today, price, amount = amount, amount_type = 'all')

                date_cur += timedelta(days=1)
        self.MA.showlog()
            

    def test_by_lstm(self, allin_rate = 0.9):
        input_model_dir = './kerasmodel_save'
        output_figure_dir = './kerasmodel_LstmNaive_balance_figures'
        if not os.path.isdir(output_figure_dir):
            os.mkdir(output_figure_dir)
        
        for modelfn in os.listdir(input_model_dir):
            print(modelfn)
            Y_predict = lstm_model2Ypredict(os.path.join(input_model_dir, modelfn), modelfn)
            date_cur = datetime.strptime('2017-01-03', '%Y-%m-%d')
            
            previous_day_info , idx = M.query(modelfn,'2016-12-30')
            previous_day_price = previous_day_info['Adj Close'][idx]
            
            for i in range( len(Y_predict) ):
                
                today = date_cur.__str__()[:10]
                _, tmpidx = M.query(modelfn, today)
                if tmpidx == -1:
                    date_cur += timedelta(days=1)
                    continue
                if today == '2020-06-02':
                    self.MA.sell_allstock('2020-06-02')
                
                QC = Y_predict[i][0]
                print(today,QC)
                if QC > 0.02:
                    print(today,QC)
                    
                    price = previous_day_price

                    budget = int(self.MA.balance * allin_rate)
                    self.MA.buy(modelfn, today, price, budget = budget, amount_type = 'budget')
                    self.MA.sell(modelfn, today, 0, amount = 0, amount_type = 'all', mode ='closeprice')


                date_cur += timedelta(days=1)

                previous_day_info , idx = M.query(modelfn,today)
                previous_day_price = previous_day_info['Adj Close'][idx]
            
            try:
                self.MA.showlog2plt(output_figure_fname = os.path.join(output_figure_dir, modelfn)+'.png')
                self.MA = MarketAccount(balance = 1000000)
            except:
                pass

def testTA():
    TA = TradeAgent()
    #TA.test_period('2018-09-18',100)
    #TA.test_by_predefined_trade('Data_buying_time.csv')
    TA.test_by_lstm()

def testM():
    A,B = M.query('2880','2017-01-04')

    print(A['Adj Close'][B])

if __name__ == '__main__':
    testTA()
    #testM()
    