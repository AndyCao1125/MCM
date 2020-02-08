import pandas as pd
import numpy as np
import seaborn as sns #热力图
import litertools 
import datetime
import matplotlib.pyplot as plt
import statsmodels.api as sm
from datetime import datetime
from statsmodels.tsa.stattolls import adfuller  #ADF检验
from statsmodels.stats.diagnostic import acorr_ljungbox  #白噪声检验
from statsmodels.graphics.tsplots import plot_acf,plot_pacf  #画图定阶

from statsmodels.tsa.armima_model import ARIMA
from statsmodels.tsa.armima_model import ARMA


df = pd.read_csv('',encoding=''utf-8', index_col='')
#1
dataframe = pd.DataFrame({'time':df[],'values':df[]})
df['time'] = pd.to_datatime(df['time'])     #将默认索引转化为时间索引
df.set_index("time", inplace=True)
#2
df.index = pd.to_datetime(df.index)
ts = df[]

#差分法实现数据平稳性
def stationarity(timeseries):
	diff1 = timeseries.diff(1).dropna()
	diff2 = diff1.diff(1)
	diff1.plot(color = 'red',title= 'diff 1',figsize=(10,4))
	diff2.plot(color = 'black',title= 'diff 2',figsize=(10,4))
	
#ADF检验
x = np.array(diff1['value'])  #看数据放一阶差分还是二阶
adftest = adfuller(x,autolag = 'AIC'=)
print (adftest)
#观察1%，5%，10%和ADF的比较，以及p-value是否接近0

#非白噪声检验
p_value = acorr_ljungbox(timeseries, lags=1)
print(p_value)

#热力度定参数（信息准则定阶AIC）
p_min=0
q_min=0
p_max=5
q_max=5
d_min=0
d_max=5
results_aic = pd.DataFrame(index=['AR{}'.format(i)\
                           for i in range (p_min,p_max+1)],\
        columns=['MA{}'.format(i) for i in range(q_min,q_max+1)])
        
for p,d,q in itertools.product(range(p_min,p_max+1),\
                               range(d_min,d_max+1),range(q_min,q_max+1)):
    if p==0 and q==0:
        results_aic.loc['AR{}'.format(p),'MA{}'.format(q)] = np.nan
        continue
    try:
        model = sm.tsa.ARIMA(timeseries,oder = (p,d,q))
        results = model.fit()
        results_aic.loc['AR{}'.format(p),'MA{}'.format(q)]=results.aic
    except:
        continue
results_aic = results_aic[results_aic.columns].astype(float)
    
fig,ax = plt.subplots(figsize=(10,8))
ax = sns.heatmap(results_aic,        #将数字显示在热力图上
                 ax = ax
                 annot = True
                 fmt = '.2f',
                 )
ax.set_title('AIC')
plt.show()


#ARMA       
def ARMA_model(train_D,train,test,order)      #train_D是原本时间数据 train是差分后的，order是定阶后的p，q
    arma_model = ARMA(train,order)   #ARMA模型
    result = arma_model.fit()
    print(result.summary())
    pred = result.predict()
    resid = result.resid      #残差
    plt.figure(figsize(12,8))
    qqplot(resid,line='q',fit=True)  #用qq图检验残差是否满足正态分布
    print('D-W test{}'.format(durbin_watson(resid.values)))  #D-W接近2，模型较好
    
    pred_one = result.predict(start= , end = ,\    #预测样本外的
                              dynamic = True)
    resid = result.resid      #残差
    plt.figure(figsize(12,8))
    qqplot(resid,line='q',fit=True)  #用qq图检验残差是否满足正态分布
    
    
#ARIMA     最后要做差分还原
def string_toDatetime(string):
    return datetime.datetime.strptime(string, "%Y-%m-%d %H:%M:%S")
    
def ARIMA_model(train_H,train,test)
    arima_model = ARIMA(train,oder(p,q,d))
    result = arima_model.fit()
    print(result.summary())
    pred = result.predict()
    
    #还原
    idx = pd.date_range(string_toDatetime(''),periods=len(),freq='D')
    pred_list = []
    for i in range(len()):
        pred_list.append(np,array(pred)[i+4])
    pred_numpy = pd.Series(np.array(pred_list),index = idx)
    pred_restored = pd.Series(np.array(train_H)[a][0],\
         index = [train_H.index[a]]).append(pred_numpy).cumsum())
    x1=np.array(pred_restored)
    x2=np.array(train_H[a:])
    y = []
    for i = range(len(pred_restored)):
        y.append(i+1)
    y = np.array(y)
    fig1 = plt.figure(num=2, figsize(10,4),dpi=80)
    plt.plot(y,x1,color = 'blue')
    plt.plot(y,x2,color = 'red')
    plt.ylim(0,0.8)
    plt.show
    
    