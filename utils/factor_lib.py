# -*- ecoding: utf-8 -*-
'''
@ModuleName: factor_lib
@Author: Chen Hao
@Time: 2023/2/9
'''

import pandas as pd
import numpy as np

pd.set_option("display.max_rows",100)
pd.set_option("display.max_columns",10)

fundamental_df=pd.read_csv('../data/first_round_train_fundamental_data.csv')
fundamental_df['code']=fundamental_df['date_time'].apply(lambda x:x[0:x.index('d')])
fundamental_df['date']=fundamental_df['date_time'].apply(lambda x:x[x.index('d')+1:])
fundamental_df['date']=fundamental_df['date'].apply(int)
fundamental_df.drop(['date_time'],axis=1,inplace=True)
fundamental_df=fundamental_df.set_index(['date','code'])

price_df = pd.read_csv('../data/market_day_freq_data.csv').set_index(['date', 'code'])


class Fundamental_data(object):
    def __init__(self,fundamental_df=fundamental_df):
        self.fundamental_df=fundamental_df
        self.turnoverRatio=fundamental_df['turnoverRatio']
        '''
        换手率；周转率（反应股票流通性指标）:成交股数/流通股数×100%
        '''
        self.transactionAmount=fundamental_df['transactionAmount']
        '''
        成交笔数，一笔可以成交多股，和成交量不同
        '''
        self.pe_ttm=fundamental_df['pe_ttm']#过去十二个月滚动市盈率
        self.pe=fundamental_df['pe']#市盈率
        '''
        市盈率＝每股市价／每股盈余
        假设某股票的市价为 24 元，而过去一年的每股盈余为 3 元，则市盈率为 24/3=8。
        该股票被视为有 8 倍的市盈率，即假设该企业以后每年净利润和去年相同的基础上，
        如果不考虑通货膨胀因素，回本期为 8 年
        市盈率越高，表示股价可能很高，而每股净利润少就可能存在泡沫；
        反之如果市盈率低，说明股价低但每股收益高，那么说明公司股价被低估。越低越好
        '''
        self.pb=fundamental_df['pb']#市净率
        '''
        每股股价与每股净资产的比率，市净率可用于股票投资分析，
        净资产定义：企业的资产减去企业的负债。也就是所有者权益，是股东可以享受到的权益。
            那么每股净资产是什么呢？我们假设这家企业倒闭，全体股东需要分钱，那么每一股可以分到的钱
            就是每股净资产。所以每股净资产代表着股票本身的内在价值。
        市净率分析：股价是股票市场上体现的，这些资产的当前的价格，是众多交易者交易的结果。
            所以很显然，市净率是一个风险指标，他表示每股价格对应了多少净资产。越低越好
        '''
        self.ps=fundamental_df['ps']#市销率
        '''
        市销率是以公司市值除以上一财年（或季度）的营业收入，或等价地，以公司股价除以每股营业收入
        市销率分析：一般对于尚未盈利的那些高成长性企业，我们是没办法用市盈率来衡量的。
            因为此时企业的利润是负的，所以市盈率也是负的。所以就要用到市销率。
            一家企业想要增加收入，就必须要扩大生产，而利润相对容易作假，比如裁员提高利润等手段
            所以市销率如果越低，相对而言就越有投资价值
        '''
        self.pcf=fundamental_df['pcf']#市现率
        '''
        市现率：股价 ÷ 每股营业现金流
        市现率分析：成长型公司，最主要的目的是帮股东创造价值，因此会把获利拿去再投资，
            这时就不适合用市盈率来估价，将股价与营收相比(市销率)，或是股价与现金流量来比较会比较恰当，
            股价与现金流量比较，就称为市现率。
        '''


class Price_data(object):
    def __init__(self, price_df = price_df):
        self.rf = 0.04  # 无风险收益率
        self.price_df = price_df
        self.open_ser = price_df['open']#当日开盘价
        self.close_ser = price_df['close']#当日收盘价
        self.high_ser = price_df['high']#当日最高价
        self.low_ser = price_df['low']#当日最低价
        self.volume_ser = price_df['volume']#当日成交量
        self.money_ser = price_df['money']#当日成交总额
        self.ret_df = self.close_ser.unstack().pct_change()


'''
Quality category factors:
Three in total
'''

class net_profit_ratio(Fundamental_data):
    '''
    Net Profit to Total Operating Income = Net Profit (TTM) / Total Operating Income (TTM)
    P/E ratio = market price per share/earnings per share, P/S ratio = market price per share/operating income per share
    -> Net profit to gross operating income = P/S ratio / P/E ratio
    '''
    def __init__(self):
        super().__init__()
    def calc(self):
        result=self.ps/self.pe
        result=pd.Series(result)
        result.rename('net_profit_ratio',inplace=True)
        return result

class cfo_to_ev(Fundamental_data):
    '''
    Net cash flows from operating activities TTM / Enterprise value. Where enterprise value = market value of the division + total liabilities - monetary capital
    P/C ratio = share price / operating cash flow per share, P/N ratio = share price / net assets per share
    -> P/N Ratio / P/C Ratio
    '''
    def __init__(self):
        super().__init__()
    def calc(self):
        result=self.pb/self.pcf
        result = pd.Series(result)
        result.rename('cfo_to_ev', inplace=True)
        return result

class net_operate_cash_flow_to_operate_income(Fundamental_data):
    '''
    Net cash flow from operating activities (TTM) / (Total operating income (TTM) - Total operating costs (TTM))
    P/C ratio = share price / operating cash flow per share, P/E ratio = market price per share / earnings per share
    -> P/E ratio / P/C ratio
    '''
    def __init__(self):
        super().__init__()
    def calc(self):
        result=self.pe/self.pcf
        result = pd.Series(result)
        result.rename('net_operate_cash_flow_to_operate_income', inplace=True)
        return result


'''
Base Factor
total of 1
'''
class net_working_capital(Price_data,Fundamental_data):
    '''
    Net Working Capital = Current Assets - Current Liabilities
    P/N ratio = share price/net asset value per share, closing price on the day, volume on the day
    -> (then closing price / P/N ratio) * then trading volume
    '''
    def __init__(self):
        Price_data.__init__(self)
        Fundamental_data.__init__(self)

    def calc(self):
        result=(self.close_ser/self.pb)*self.volume_ser
        result = pd.Series(result)
        result.rename('net_working_capital', inplace=True)
        return result



'''
Momentum factors
total of 23
'''
class arron_up_25(Price_data,Fundamental_data):
    '''
    Aroon(rise)=[(days of calculation period - days after highest price)/days of calculation period]*100

    '''
    def __init__(self):
        Price_data.__init__(self)
        Fundamental_data.__init__(self)


    def calc(self):
        highest_series=1001-self.close_ser.unstack().idxmax()
        highest_series.name='highest_series'
        self.price_df=pd.merge(self.price_df,highest_series,left_index=True,right_index=True)
        #index.get_level_values可以取出特定的index的值
        self.price_df['Aroon']=(((1001-self.price_df.index.get_level_values('date'))-(self.price_df['highest_series']))/(1001-self.price_df.index.get_level_values('date')))*100
        return self.price_df['Aroon']

class arron_down_25(Price_data,Fundamental_data):
    '''
    Aroon(Decline) = [(days of calculation period - days after minimum price)/days of calculation period]*100

    '''
    def __init__(self):
        Price_data.__init__(self)
        Fundamental_data.__init__(self)

    def calc(self):
        lowest_series = 1001 - self.close_ser.unstack().idxmin()
        lowest_series.name = 'lowest_series'
        self.price_df = pd.merge(self.price_df, lowest_series, left_index=True,
                                 right_index=True)  # 这个地方根据index进行merge的时候会自动广播
        # index.get_level_values可以取出特定的index的值
        self.price_df['Aroon'] = (((1001 - self.price_df.index.get_level_values('date')) - (
        self.price_df['lowest_series'])) / (1001 - self.price_df.index.get_level_values('date'))) * 100
        return self.price_df['Aroon']

class BBIC(Price_data):
    '''
    1. 3-day average = sum of 3-day closing prices / 3
    2. 6-Day Average = Sum of 6-Day Closing Prices / 6
    3. 12-Day Average = Sum of 12-Day Closing Prices/12
    4. 24-Day Average = Sum of 24-Day Closing Prices / 24
    5. BBI = (3-Day Average + 6-Day Average + 12-Day Average + 24-Day Average)/4
    '''
    def __init__(self):
        super().__init__()
    def calc(self):
        avreage_3=self.close_ser.unstack().rolling(window=3).mean()
        avreage_6=self.close_ser.unstack().rolling(window=6).mean()
        avreage_12=self.close_ser.unstack().rolling(window=12).mean()
        avreage_24=self.close_ser.unstack().rolling(window=24).mean()
        BBIC_unstack=(avreage_3+avreage_6+avreage_12+avreage_24)/4
        BBIC=BBIC_unstack.stack(dropna=False)
        result = pd.Series(BBIC)
        result.rename('BBIC', inplace=True)
        return result

class bear_power(Price_data):
    '''
    (Lowest price - EMA(close,13)) / close
    '''
    def __init__(self):
        super().__init__()
    def calc(self):
        EMA=self.close_ser.unstack().ewm(span=12,adjust=False).mean()
        result=((self.low_ser.unstack()-EMA)/self.close_ser.unstack()).stack(dropna=False)
        result = pd.Series(result)
        result.rename('bear_power', inplace=True)
        return result

class bull_power(Price_data):
    '''
    (Highest price - EMA(close,13)) / close
    '''

    def __init__(self):
        super().__init__()

    def calc(self):
        EMA = self.close_ser.unstack().ewm(span=12, adjust=False).mean()
        result=((self.high_ser.unstack() - EMA) / self.close_ser.unstack()).stack(dropna=False)
        result = pd.Series(result)
        result.rename('bull_power', inplace=True)
        return result

class BIAS5(Price_data):
    '''
    (Closing price - N-day simple average of closing prices) / N-day simple average of closing prices * 100, where n is taken as 5
    '''
    def __init__(self):
        super().__init__()

    def calc(self):
        average_close_unstack=self.price_df['close'].unstack().rolling(window=5).mean()
        close_unstack=self.price_df['close'].unstack()
        bias5_unstack=(close_unstack-average_close_unstack)/average_close_unstack*100
        bias5=bias5_unstack.stack(dropna=False)
        result = pd.Series(bias5)
        result.rename('bias5', inplace=True)
        return result

class BIAS10(Price_data):
    '''
    (Closing price - N-day simple average of closing prices) / N-day simple average of closing prices * 100, where n is taken as 10
    '''
    def __init__(self):
        super().__init__()
    def calc(self):
        average_close_unstack = self.price_df['close'].unstack().rolling(window=10).mean()
        close_unstack = self.price_df['close'].unstack()
        bias10_unstack = (close_unstack - average_close_unstack) / average_close_unstack * 100
        bias10 = bias10_unstack.stack(dropna=False)
        result = pd.Series(bias10)
        result.rename('bias10', inplace=True)
        return result

class BIAS20(Price_data):
    '''
    (Closing price - N-day simple average of closing prices) / N-day simple average of closing prices * 100, where n is taken as 20
    '''
    def __init__(self):
        super().__init__()
    def calc(self):
        average_close_unstack = self.price_df['close'].unstack().rolling(window=20).mean()
        close_unstack = self.price_df['close'].unstack()
        bias20_unstack = (close_unstack - average_close_unstack) / average_close_unstack * 100
        bias20 = bias20_unstack.stack(dropna=False)
        result = pd.Series(bias20)
        result.rename('bias20', inplace=True)
        return result

class BIAS60(Price_data):
    '''
    (Closing price - N-day simple average of closing prices) / N-day simple average of closing prices * 100, where n is taken as 60
    '''
    def __init__(self):
        super().__init__()
    def calc(self):
        average_close_unstack = self.price_df['close'].unstack().rolling(window=60).mean()
        close_unstack = self.price_df['close'].unstack()
        bias60_unstack = (close_unstack - average_close_unstack) / average_close_unstack * 100
        bias60 = bias60_unstack.stack(dropna=False)
        result = pd.Series(bias60)
        result.rename('bias60', inplace=True)
        return result

class CR20(Price_data):
    '''
    ①Median price = highest price 1 day ago + lowest price / 2
    ②Up value = today's high price - previous day's mid price (negative values are recorded as 0)
    ③Down value = previous day's mid price - today's low (negative values are recorded as 0)
    ④Multi-side strength = sum of 20 days' upside, short-side strength = sum of 20 days' downside
    ⑤ CR = (multi-side strength ÷ short-side strength) x 100
    '''
    def __init__(self):
        super().__init__()
    def calc(self):
        middle_unstack=((self.high_ser.unstack().shift()+self.low_ser.unstack().shift())/2)
        rise_unstack=self.high_ser.unstack()-middle_unstack
        rise_unstack[rise_unstack<0]=0
        down_unstack=middle_unstack-self.low_ser.unstack()
        down_unstack[rise_unstack<0]=0
        multi_strength=rise_unstack.rolling(window=20).sum()
        empty_strength=down_unstack.rolling(window=20).sum()
        CR_unstack=(multi_strength/empty_strength)*100
        CR=CR_unstack.stack(dropna=False)
        result = pd.Series(CR)
        result.rename('CR', inplace=True)
        return result

class Price1M(Price_data):
    '''
    Day's closing price / mean(closing price in the past month (21 days)) - 1
    '''
    def __init__(self):
        super().__init__()
    def calc(self):
        average_close=self.close_ser.unstack().rolling(window=21).mean()
        Price1M_unstack=(self.close_ser.unstack()/average_close)-1
        Price1M=Price1M_unstack.stack(dropna=False)
        result = pd.Series(Price1M)
        result.rename('Price1M', inplace=True)
        return result

class Price3M(Price_data):
    '''
    Day's closing price / mean(closing price in the past month (61 days)) - 1
    '''
    def __init__(self):
        super().__init__()
    def calc(self):
        average_close=self.close_ser.unstack().rolling(window=61).mean()
        Price3M_unstack=(self.close_ser.unstack()/average_close)-1
        Price3M=Price3M_unstack.stack(dropna=False)
        result = pd.Series(Price3M)
        result.rename('Price3M', inplace=True)
        return result


class Price1Y(Price_data):
    '''
    Day's closing price / mean (closing price in the past month (250 days)) -1
    '''
    def __init__(self):
        super().__init__()
    def calc(self):
        average_close=self.close_ser.unstack().rolling(window=250).mean()
        Price1Y_unstack=(self.close_ser.unstack()/average_close)-1
        Price1Y=Price1Y_unstack.stack(dropna=False)
        result = pd.Series(Price1Y)
        result.rename('Price1Y', inplace=True)
        return result

class ROC12(Price_data):
    '''
    ①AX=Today's closing price - 12 days ago's closing price
    ②BX=close price 12 days ago
    ③ROC=AX/BX*100
    '''
    def __init__(self):
        super().__init__()
    def calc(self):
        AX=self.close_ser.unstack()-self.close_ser.unstack().shift(periods=12)
        BX=self.close_ser.unstack().shift(periods=12)
        ROC_unstack=AX/BX*100
        ROC=ROC_unstack.stack(dropna=False)
        result = pd.Series(ROC)
        result.rename('ROC12', inplace=True)
        return result

class ROC120(Price_data):
    '''
    ①AX=Today's closing price - 120 days ago's closing price
    ②BX=close price 120 days ago
    ③ROC=AX/BX*100
    '''
    def __init__(self):
        super().__init__()
    def calc(self):
        AX=self.close_ser.unstack()-self.close_ser.unstack().shift(periods=120)
        BX=self.close_ser.unstack().shift(periods=120)
        ROC_unstack=AX/BX*100
        ROC=ROC_unstack.stack(dropna=False)
        result = pd.Series(ROC)
        result.rename('ROC120', inplace=True)
        return result

class ROC20(Price_data):
    '''
    ①AX=Today's closing price - 20 days ago's closing price
    ②BX=closing price 20 days ago
    ③ROC=AX/BX*100
    '''
    def __init__(self):
        super().__init__()
    def calc(self):
        AX=self.close_ser.unstack()-self.close_ser.unstack().shift(periods=20)
        BX=self.close_ser.unstack().shift(periods=20)
        ROC_unstack=AX/BX*100
        ROC=ROC_unstack.stack(dropna=False)
        result = pd.Series(ROC)
        result.rename('ROC20', inplace=True)
        return result

class ROC6(Price_data):
    '''
    ①AX=Today's closing price - 6 days ago's closing price
    ②BX=close price 6 days ago
    ③ROC=AX/BX*100
    '''
    def __init__(self):
        super().__init__()
    def calc(self):
        AX=self.close_ser.unstack()-self.close_ser.unstack().shift(periods=6)
        BX=self.close_ser.unstack().shift(periods=6)
        ROC_unstack=AX/BX*100
        ROC=ROC_unstack.stack(dropna=False)
        result = pd.Series(ROC)
        result.rename('ROC6', inplace=True)
        return result

class ROC60(Price_data):
    '''
    ①AX=Today's closing price - 60 days ago's closing price
    ②BX=Closing price 60 days ago
    ③ROC=AX/BX*100
    '''
    def __init__(self):
        super().__init__()
    def calc(self):
        AX=self.close_ser.unstack()-self.close_ser.unstack().shift(periods=60)
        BX=self.close_ser.unstack().shift(periods=60)
        ROC_unstack=AX/BX*100
        ROC=ROC_unstack.stack(dropna=False)
        result = pd.Series(ROC)
        result.rename('ROC60', inplace=True)
        return result

class single_day_VPT(Price_data):
    '''
    (Today's closing price - yesterday's closing price) / yesterday's closing price * current day's volume # (compounding method is based on current day's ex-weighting)    '''
    def __init__(self):
        super().__init__()
    def calc(self):
        upper=self.close_ser.unstack()-self.close_ser.unstack().shift()
        down=self.close_ser.unstack().shift()*self.volume_ser.unstack()
        result=(upper/down).stack(dropna=False)
        result = pd.Series(result)
        result.rename('single_day_VPT', inplace=True)
        return result

class single_day_VPT_12(Price_data):
    '''
    MA(single_day_VPT, 12)
    '''
    def __init__(self):
        super().__init__()
    def calc(self):
        upper=self.close_ser.unstack()-self.close_ser.unstack().shift()
        down=self.close_ser.unstack().shift()*self.volume_ser.unstack()
        single_day_VPT=upper/down
        single_day_VPT_12_unstack=single_day_VPT.rolling(window=12).mean()
        result=single_day_VPT_12_unstack.stack(dropna=False)
        result = pd.Series(result)
        result.rename('single_day_VPT_12', inplace=True)
        return result


class single_day_VPT_6(Price_data):
    '''
    MA(single_day_VPT, 6)
    '''
    def __init__(self):
        super().__init__()
    def calc(self):
        upper=self.close_ser.unstack()-self.close_ser.unstack().shift()
        down=self.close_ser.unstack().shift()*self.volume_ser.unstack()
        single_day_VPT=upper/down
        single_day_VPT_6_unstack=single_day_VPT.rolling(window=6).mean()
        result=single_day_VPT_6_unstack.stack(dropna=False)
        result = pd.Series(result)
        result.rename('single_day_VPT_6', inplace=True)
        return result

class TRIX10(Price_data):
    '''
    MTR=10-day exponential moving average of the 10-day exponential moving average of the closing price (find ema10 three times).
    TRIX=(MTR - MTR before 1 day)/MTR before 1 day*100
    '''
    def __init__(self):
        super().__init__()
    def calc(self):
        ewm1=self.close_ser.unstack().ewm(span=10,adjust=False).mean()
        ewm2=ewm1.ewm(span=10,adjust=False).mean()
        MTR=ewm2.ewm(span=10,adjust=False).mean()
        TRIX=(MTR-MTR.shift())/MTR.shift()*100
        result=TRIX.stack(dropna=False)
        result = pd.Series(result)
        result.rename('TRIX10', inplace=True)
        return result

class TRIX5(Price_data):
    '''
        MTR=5-day exponential moving average of the 5-day exponential moving average of the closing price (find ema5 three times).
        TRIX=(MTR - MTR before 1 day)/MTR before 1 day*100
    '''

    def __init__(self):
        super().__init__()

    def calc(self):
        ewm1 = self.close_ser.unstack().ewm(span=5, adjust=False).mean()
        ewm2 = ewm1.ewm(span=5, adjust=False).mean()
        MTR = ewm2.ewm(span=5, adjust=False).mean()
        TRIX = (MTR - MTR.shift()) / MTR.shift() * 100
        result=TRIX.stack(dropna=False)
        result = pd.Series(result)
        result.rename('TRIX5', inplace=True)
        return result

'''
Technology factor
total of 14
'''
class boll_down(Price_data):
    '''
    (MA(CLOSE,M)-2*STD(CLOSE,M)) / closing price; M=20
    '''
    def __init__(self):
        super().__init__()
    def calc(self):
        close_unstack=self.close_ser.unstack()
        boll_down_unstack=(close_unstack.rolling(window=20).mean()-2*close_unstack.rolling(window=20).std())/close_unstack
        result=boll_down_unstack.stack(dropna=False)
        result = pd.Series(result)
        result.rename('boll_down', inplace=True)
        return result

class boll_up(Price_data):
    '''
    (MA(CLOSE,M)+2*STD(CLOSE,M)) / closing price; M=20
    '''
    def __init__(self):
        super().__init__()
    def calc(self):
        close_unstack=self.close_ser.unstack()
        boll_up_unstack=(close_unstack.rolling(window=20).mean()+2*close_unstack.rolling(window=20).std())/close_unstack
        result=boll_up_unstack.stack(dropna=False)
        result = pd.Series(result)
        result.rename('boll_up', inplace=True)
        return result

class EMA5(Price_data):
    '''
    5-Day Exponential Moving Average / Today's Closing Price
    '''
    def __init__(self):
        super().__init__()
    def calc(self):
        close_unstack=self.close_ser.unstack()
        EMA5_unstack=close_unstack.ewm(span=5,adjust=False).mean()/close_unstack
        result=EMA5_unstack.stack(dropna=False)
        result = pd.Series(result)
        result.rename('EMA5', inplace=True)
        return result

class EMAC10(Price_data):
    '''
    10-Day Exponential Moving Average / Today's Closing Price
    '''
    def __init__(self):
        super().__init__()
    def calc(self):
        close_unstack=self.close_ser.unstack()
        EMAC10_unstack=close_unstack.ewm(span=10,adjust=False).mean()/close_unstack
        result=EMAC10_unstack.stack(dropna=False)
        result = pd.Series(result)
        result.rename('EMAC10', inplace=True)
        return result

class EMAC12(Price_data):
    '''
    12-Day Exponential Moving Average / Today's Closing Price
    '''
    def __init__(self):
        super().__init__()
    def calc(self):
        close_unstack=self.close_ser.unstack()
        EMAC12_unstack=close_unstack.ewm(span=12,adjust=False).mean()/close_unstack
        result=EMAC12_unstack.stack(dropna=False)
        result = pd.Series(result)
        result.rename('EMAC12', inplace=True)
        return result

class EMAC20(Price_data):
    '''
    20-Day Exponential Moving Average / Today's Closing Price
    '''
    def __init__(self):
        super().__init__()
    def calc(self):
        close_unstack=self.close_ser.unstack()
        EMAC20_unstack=close_unstack.ewm(span=20,adjust=False).mean()/close_unstack
        result=EMAC20_unstack.stack(dropna=False)
        result = pd.Series(result)
        result.rename('EMAC20', inplace=True)
        return result

class EMAC26(Price_data):
    '''
    26-Day Exponential Moving Average / Today's Closing Price
    '''
    def __init__(self):
        super().__init__()
    def calc(self):
        close_unstack=self.close_ser.unstack()
        EMAC26_unstack=close_unstack.ewm(span=26,adjust=False).mean()/close_unstack
        result=EMAC26_unstack.stack(dropna=False)
        result = pd.Series(result)
        result.rename('EMAC26', inplace=True)
        return result

class EMAC120(Price_data):
    '''
    120-Day Exponential Moving Average / Today's Closing Price
    '''
    def __init__(self):
        super().__init__()
    def calc(self):
        close_unstack=self.close_ser.unstack()
        EMAC120_unstack=close_unstack.ewm(span=120,adjust=False).mean()/close_unstack
        result=EMAC120_unstack.stack(dropna=False)
        result = pd.Series(result)
        result.rename('EMAC120', inplace=True)
        return result

class MAC5(Price_data):
    '''
        5 Day Moving Average / Today's Closing Price
        '''

    def __init__(self):
        super().__init__()

    def calc(self):
        close_unstack = self.close_ser.unstack()
        MAC5_unstack = close_unstack.rolling(window=5).mean() / close_unstack
        result=MAC5_unstack.stack(dropna=False)
        result = pd.Series(result)
        result.rename('MAC5', inplace=True)
        return result

class MAC10(Price_data):
    '''
        10 Day Moving Average / Today's Closing Price
        '''

    def __init__(self):
        super().__init__()

    def calc(self):
        close_unstack = self.close_ser.unstack()
        MAC10_unstack = close_unstack.rolling(window=10).mean() / close_unstack
        result=MAC10_unstack.stack(dropna=False)
        result = pd.Series(result)
        result.rename('MAC10', inplace=True)
        return result

class MAC20(Price_data):
    '''
        20 Day Moving Average / Today's Closing Price
        '''

    def __init__(self):
        super().__init__()

    def calc(self):
        close_unstack = self.close_ser.unstack()
        MAC20_unstack = close_unstack.rolling(window=20).mean() / close_unstack
        result=MAC20_unstack.stack(dropna=False)
        result = pd.Series(result)
        result.rename('MAC20', inplace=True)
        return result

class MAC60(Price_data):
    '''
        60 Day Moving Average / Today's Closing Price
        '''

    def __init__(self):
        super().__init__()

    def calc(self):
        close_unstack = self.close_ser.unstack()
        MAC60_unstack = close_unstack.rolling(window=60).mean() / close_unstack
        result=MAC60_unstack.stack(dropna=False)
        result = pd.Series(result)
        result.rename('MAC60', inplace=True)
        return result

class MAC120(Price_data):
    '''
        120 Day Moving Average / Today's Closing Price
        '''

    def __init__(self):
        super().__init__()

    def calc(self):
        close_unstack = self.close_ser.unstack()
        MAC120_unstack = close_unstack.rolling(window=120).mean() / close_unstack
        result=MAC120_unstack.stack(dropna=False)
        result = pd.Series(result)
        result.rename('MAC120', inplace=True)
        return result

class MACDC(Price_data):
    '''
    MACD(SHORT=12, LONG=26, MID=9) / Today's Closing Price
    '''
    def __init__(self):
        super(MACDC, self).__init__()
    def calc(self):
        short=self.close_ser.unstack().ewm(span=12,adjust=False).mean()
        long=self.close_ser.unstack().ewm(span=26,adjust=False).mean()
        diff=short-long
        dea=diff.ewm(span=9,adjust=False).mean()
        MACD_unstack=2*(diff-dea)
        result=MACD_unstack.stack(dropna=False)
        result = pd.Series(result)
        result.rename('MACDC', inplace=True)
        return result


'''
Per share factor
total of 5
'''
class total_operating_revenue_per_share(Price_data,Fundamental_data):
    '''
    Gross operating income per share
    Price to sales ratio = Market price per share / Operating income per share
    '''
    def __init__(self):
        Price_data.__init__(self)
        Fundamental_data.__init__(self)
    def calc(self):
        income_per_set=(self.close_ser/self.ps).unstack()
        result=income_per_set.sum()
        result = pd.Series(result)
        result.rename('total_operating_revenue_per_share', inplace=True)
        return result

class operating_revenue_per_share(Price_data,Fundamental_data):
    '''
        Operating income per share
        Price to sales ratio = Market price per share / Operating income per share
        '''

    def __init__(self):
        Price_data.__init__(self)
        Fundamental_data.__init__(self)

    def calc(self):
        income_per_set = (self.close_ser / self.ps)
        result=income_per_set
        result = pd.Series(result)
        result.rename('operating_revenue_per_share', inplace=True)
        return result

class net_asset_per_share(Price_data,Fundamental_data):
    '''
    Net assets per share
    P/N ratio = share price / net assets per share
    '''
    def __init__(self):
        Price_data.__init__(self)
        Fundamental_data.__init__(self)

    def calc(self):
        result=self.close_ser/self.pb
        result = pd.Series(result)
        result.rename('net_asset_per_share', inplace=True)
        return result

class net_operate_cash_flow_per_share(Price_data,Fundamental_data):
    '''
        Net cash flow from operating activities per share
        Price-to-cash ratio = share price / operating cash flow per share
        '''

    def __init__(self):
        Price_data.__init__(self)
        Fundamental_data.__init__(self)
    def calc(self):
        result=self.close_ser/self.pcf
        result = pd.Series(result)
        result.rename('net_operate_cash_flow_per_share', inplace=True)
        return result

class operating_profit_per_share(Price_data,Fundamental_data):
    '''
    Operating profit per share
    P/E ratio = Market price per share / Earnings per share
    '''

    def __init__(self):
        Price_data.__init__(self)
        Fundamental_data.__init__(self)
    def calc(self):
        result=self.close_ser/self.pe
        result = pd.Series(result)
        result.rename('operating_profit_per_share', inplace=True)
        return result


if __name__=="__main__":
    #get all the subclass in order to get all the factors
    subclasses = set()
    for subclass in Fundamental_data.__subclasses__():
        subclasses.add(subclass)
    for subclass in Price_data.__subclasses__():
        subclasses.add(subclass)

    subclasses=list(subclasses)
    df=subclasses[0]().calc()
    for i in range(1,len(subclasses)):
        df=pd.merge(df,subclasses[i]().calc(),left_index=True,right_index=True)
    df.to_csv('../data/factors_of_trianing.csv')


























