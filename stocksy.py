import yfinance as yf
import pandas as pd
import datetime
import requests as r
from bs4 import BeautifulSoup as bs
#import pickle
import numpy as np
from scipy import stats
import streamlit as st
import streamlit.components.v1 as components


st.set_page_config(
     page_title="Stocksy App",
     page_icon="🧊",
     layout="wide"
     )

endDate = datetime.datetime.now().date()
days = 100
d = datetime.timedelta(days = days)
startDate = endDate - d
startDate = startDate#.date()
print("Getting {} days worth of data from {} to {}".format(days, startDate, endDate))

@st.cache_data()
def get_stocks_data():
	df = yf.download(stocks, start=startDate, end=endDate,group_by="ticker")
	return df


stocks = ['MMM', 'ABT', 'ABBV', 'ACN', 'ATVI', 'AYI', 'ADBE', 'AMD', 'AAP', 'AES', 'AET', 'AMG', 'AFL', 'A', 'APD', 'AKAM', 'ALK', 'ALB', 'ARE', 'ALXN', 'ALGN', 'ALLE', 'AGN', 'ADS', 'LNT', 'ALL', 'GOOGL', 'GOOG', 'MO', 'AMZN', 'AEE', 'AAL', 'AEP', 'AXP', 'AIG', 'AMT', 'AWK', 'AMP', 'ABC', 'AME', 'AMGN', 'APH', 'APC', 'ADI', 'ANDV', 'ANSS', 'ANTM', 'AON', 'AOS', 'APA', 'AIV', 'AAPL', 'AMAT', 'ADM', 'ARNC', 'AJG', 'AIZ', 'T', 'ADSK', 'ADP', 'AZO', 'AVB', 'AVY', 'BHGE', 'BLL', 'BAC', 'BK', 'BCR', 'BAX', 'BBT', 'BDX', 'BRK.B', 'BBY', 'BIIB', 'BLK', 'HRB', 'BA', 'BWA', 'BXP', 'BSX', 'BHF', 'BMY', 'AVGO', 'BF.B', 'CHRW', 'CA', 'COG', 'CDNS', 'CPB', 'COF', 'CAH', 'CBOE', 'KMX', 'CCL', 'CAT', 'CBG', 'CBS', 'CELG', 'CNC', 'CNP', 'CTL', 'CERN', 'CF', 'SCHW', 'CHTR', 'CHK', 'CVX', 'CMG', 'CB', 'CHD', 'CI', 'XEC', 'CINF', 'CTAS', 'CSCO', 'C', 'CFG', 'CTXS', 'CLX', 'CME', 'CMS', 'KO', 'CTSH', 'CL', 'CMCSA', 'CMA', 'CAG', 'CXO', 'COP', 'ED', 'STZ', 'COO', 'GLW', 'COST', 'COTY', 'CCI', 'CSRA', 'CSX', 'CMI', 'CVS', 'DHI', 'DHR', 'DRI', 'DVA', 'DE', 'DLPH', 'DAL', 'XRAY', 'DVN', 'DLR', 'DFS', 'DISCA', 'DISCK', 'DISH', 'DG', 'DLTR', 'D', 'DOV', 'DWDP', 'DPS', 'DTE', 'DRE', 'DUK', 'DXC', 'ETFC', 'EMN', 'ETN', 'EBAY', 'ECL', 'EIX', 'EW', 'EA', 'EMR', 'ETR', 'EVHC', 'EOG', 'EQT', 'EFX', 'EQIX', 'EQR', 'ESS', 'EL', 'ES', 'RE', 'EXC', 'EXPE', 'EXPD', 'ESRX', 'EXR', 'XOM', 'FFIV', 'FB', 'FAST', 'FRT', 'FDX', 'FIS', 'FITB', 'FE', 'FISV', 'FLIR', 'FLS', 'FLR', 'FMC', 'FL', 'F', 'FTV', 'FBHS', 'BEN', 'FCX', 'GPS', 'GRMN', 'IT', 'GD', 'GE', 'GGP', 'GIS', 'GM', 'GPC', 'GILD', 'GPN', 'GS', 'GT', 'GWW', 'HAL', 'HBI', 'HOG', 'HRS', 'HIG', 'HAS', 'HCA', 'HCP', 'HP', 'HSIC', 'HSY', 'HES', 'HPE', 'HLT', 'HOLX', 'HD', 'HON', 'HRL', 'HST', 'HPQ', 'HUM', 'HBAN', 'IDXX', 'INFO', 'ITW', 'ILMN', 'IR', 'INTC', 'ICE', 'IBM', 'INCY', 'IP', 'IPG', 'IFF', 'INTU', 'ISRG', 'IVZ', 'IRM', 'JEC', 'JBHT', 'SJM', 'JNJ', 'JCI', 'JPM', 'JNPR', 'KSU', 'K', 'KEY', 'KMB', 'KIM', 'KMI', 'KLAC', 'KSS', 'KHC', 'KR', 'LB', 'LLL', 'LH', 'LRCX', 'LEG', 'LEN', 'LUK', 'LLY', 'LNC', 'LKQ', 'LMT', 'L', 'LOW', 'LYB', 'MTB', 'MAC', 'M', 'MRO', 'MPC', 'MAR', 'MMC', 'MLM', 'MAS', 'MA', 'MAT', 'MKC', 'MCD', 'MCK', 'MDT', 'MRK', 'MET', 'MTD', 'MGM', 'KORS', 'MCHP', 'MU', 'MSFT', 'MAA', 'MHK', 'TAP', 'MDLZ', 'MON', 'MNST', 'MCO', 'MS', 'MOS', 'MSI', 'MYL', 'NDAQ', 'NOV', 'NAVI', 'NTAP', 'NFLX', 'NWL', 'NFX', 'NEM', 'NWSA', 'NWS', 'NEE', 'NLSN', 'NKE', 'NI', 'NBL', 'JWN', 'NSC', 'NTRS', 'NOC', 'NCLH', 'NRG', 'NUE', 'NVDA', 'ORLY', 'OXY', 'OMC', 'OKE', 'ORCL', 'PCAR', 'PKG', 'PH', 'PDCO', 'PAYX', 'PYPL', 'PNR', 'PBCT', 'PEP', 'PKI', 'PRGO', 'PFE', 'PCG', 'PM', 'PSX', 'PNW', 'PXD', 'PNC', 'RL', 'PPG', 'PPL', 'PX', 'PCLN', 'PFG', 'PG', 'PGR', 'PLD', 'PRU', 'PEG', 'PSA', 'PHM', 'PVH', 'QRVO', 'PWR', 'QCOM', 'DGX', 'Q', 'RRC', 'RJF', 'RTN', 'O', 'RHT', 'REG', 'REGN', 'RF', 'RSG', 'RMD', 'RHI', 'ROK', 'COL', 'ROP', 'ROST', 'RCL', 'CRM', 'SBAC', 'SCG', 'SLB', 'SNI', 'STX', 'SEE', 'SRE', 'SHW', 'SIG', 'SPG', 'SWKS', 'SLG', 'SNA', 'SO', 'LUV', 'SPGI', 'SWK', 'SBUX', 'STT', 'SRCL', 'SYK', 'STI', 'SYMC', 'SYF', 'SNPS', 'SYY', 'TROW', 'TPR', 'TGT', 'TEL', 'FTI', 'TXN', 'TXT', 'TMO', 'TIF', 'TWX', 'TJX', 'TMK', 'TSS', 'TSCO', 'TDG', 'TRV', 'TRIP', 'FOXA', 'FOX', 'TSN', 'UDR', 'ULTA', 'USB', 'UA', 'UAA', 'UNP', 'UAL', 'UNH', 'UPS', 'URI', 'UTX', 'UHS', 'UNM', 'VFC', 'VLO', 'VAR', 'VTR', 'VRSN', 'VRSK', 'VZ', 'VRTX', 'VIAB', 'V', 'VNO', 'VMC', 'WMT', 'WBA', 'DIS', 'WM', 'WAT', 'WEC', 'WFC', 'HCN', 'WDC', 'WU', 'WRK', 'WY', 'WHR', 'WMB', 'WLTW', 'WYN', 'WYNN', 'XEL', 'XRX', 'XLNX', 'XL', 'XYL', 'YUM', 'ZBH', 'ZION', 'ZTS']
#stocks = ['TSLA','NVDA','AAPL']
@st.cache_data()
def save_sp500_tickers():
    resp = r.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs(resp.text, 'lxml')
    table = soup.find('table', {'class': 'wikitable sortable'})
    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text
        tickers.append(ticker)
    # with open("sp500tickers.pickle","wb") as f:
    #     pickle.dump(tickers,f)
    tickers = list(map(lambda s: s.strip(), tickers))

    return tickers


def get_results(df):

    df['ema13_close']= df.groupby("Symbol").apply(lambda x: x["Close"].ewm(span=13).mean()).reset_index()['Close']
    df['ema20_close']= df.groupby("Symbol").apply(lambda x: x["Close"].ewm(span=21).mean()).reset_index()['Close']
    df['ema48_close']= df.groupby("Symbol").apply(lambda x: x["Close"].ewm(span=48).mean()).reset_index()['Close']

    # df['ema13_close'] = df.Close.ewm(span=13, adjust=False).mean()
    # df['ema20_close'] = df.Close.ewm(span=20, adjust=False).mean()
    # df['ema48_close'] = df.Close.ewm(span=48, adjust=False).mean()
    df['diff_close'] = df['ema13_close'] - df['ema20_close']
    df['ppchange_close_1320'] = (df['ema13_close']/df['ema20_close'])-1
    df['ppchange_close_1348'] = (df['ema13_close']/df['ema48_close'])-1

    df['ema13_vol']= df.groupby("Symbol").apply(lambda x: x["Vol"].ewm(span=13).mean()/1000000).reset_index()['Vol']
    df['ema20_vol']= df.groupby("Symbol").apply(lambda x: x["Vol"].ewm(span=21).mean()/1000000).reset_index()['Vol']
    df['ema48_vol']= df.groupby("Symbol").apply(lambda x: x["Vol"].ewm(span=48).mean()/1000000).reset_index()['Vol']

    # df['ema13_vol'] = df.Vol.ewm(span=13, adjust=False).mean()/1000000
    # df['ema20_vol'] = df.Vol.ewm(span=20, adjust=False).mean()/1000000
    # df['ema48_vol'] = df.Vol.ewm(span=48, adjust=False).mean()/1000000
    df['diff_vol_13_20'] = df['ema13_vol'] - df['ema20_vol']
    df['diff_vol_13_48'] = df['ema13_vol'] - df['ema48_vol']
    df['ppchange_vol_13_20'] = (df['ema13_vol']/df['ema20_vol'])-1
    df['ppchange_vol_13_48'] = (df['ema13_vol']/df['ema48_vol'])-1

#     df = df.set_index('date2')
#     df = df.sort_index()
#     df= df.sort_values(by=['Symbol','date2'])

    # Add total prior converges to the stock

    df['Data_lagged'] = df.groupby(['Symbol'])['ppchange_close_1320'].shift(1)
    conditions = [ (df['Data_lagged'] < 0) & (df['ppchange_close_1320'] > 0)]
    choices = [1]
    df['flag'] = np.select(conditions,choices)

    df['total_converge1320'] = df.groupby('Symbol')['flag'].transform('sum')

    l14d= df.groupby('Symbol').tail(14).groupby('Symbol')['flag'].sum().reset_index()
    l14d = l14d.rename(columns={'flag':'l14dconverge1320'})
    df = df.join(l14d.set_index('Symbol'), on='Symbol')


    df = df.reset_index()
    # adding in slope of curve
    df['date2'] = pd.to_datetime(df['Date'])

    #Calculate the slope of the line based on the last x number of days
    df2 = df[df["date2"] >= (pd.to_datetime('today') - pd.Timedelta(days=13))]
    df2['date_ordinal'] = pd.to_datetime(df['date2']).map(datetime.datetime.toordinal)
    slope = pd.DataFrame(df2.groupby('Symbol').apply(lambda v: stats.linregress(v.date_ordinal, v.Close)[0]),columns=['slope_13'])

    #Calculate the slope of the line based on the last x number of days
    df2 = df[df["date2"] >= (pd.to_datetime('today') - pd.Timedelta(days=21))]
    df2['date_ordinal'] = pd.to_datetime(df['date2']).map(datetime.datetime.toordinal)
    slope21 = pd.DataFrame(df2.groupby('Symbol').apply(lambda v: stats.linregress(v.date_ordinal, v.Close)[0]),columns=['slope_21'])

    df2 = df[df["date2"] >= (pd.to_datetime('today') - pd.Timedelta(days=5))]
    df2['date_ordinal'] = pd.to_datetime(df['date2']).map(datetime.datetime.toordinal)
    slope2 = pd.DataFrame(df2.groupby('Symbol').apply(lambda v: stats.linregress(v.date_ordinal, v.Close)[0]),columns=['slope_5'])

    #df['date_ordinal'] = pd.to_datetime(df['date2']).map(datetime.toordinal)
    #slope, intercept, r_value, p_value, std_err = stats.linregress(df['date_ordinal'], df['Open'])
    #df['slope'] = slope

    #Join the dataframe on the symbol value
    df = df.join(slope,on='Symbol')
    df = df.join(slope21,on='Symbol')
    df = df.join(slope2,on='Symbol')

    #normalize slope
    df['slope13_normal'] = df['slope_13']/df['Close']
    df['slope21_normal'] = df['slope_21']/df['Close']
    df['slope5_normal'] = df['slope_5']/df['Close']

    #Create flags
    df['vol_13v20'] = df['ppchange_vol_13_20'].apply(lambda x: 1 if x >= 0.05 else 0)
    df['vol_13v48'] = df['ppchange_vol_13_48'].apply(lambda x: 1 if x >= 0.05 else 0)

    df['slope13'] = df['slope13_normal'].apply(lambda x: 1 if x >= 0 else 0)
    df['slope5'] = df['slope5_normal'].apply(lambda x: 1 if x >= 0 else 0)
    df['slope21'] = df['slope21_normal'].apply(lambda x: 1 if x >= 0 else 0)

    df['ema13v21'] = df['ppchange_close_1320'].apply(lambda x: 1 if x >= -0.02 and x <= 0.05 else 0)

    df['closevopen'] = df.apply(lambda x : 1 if x['Close'] >= x['Open'] else 0, axis =1)
    df['closevema'] = df.apply(lambda x : 1 if x['Close'] >= x['ema13_close'] else 0, axis =1)

    df['Total_Score'] = df['slope13'] + df['slope5'] +df['slope21']+df['vol_13v20']+df['vol_13v48']+ df['ema13v21']+df['closevopen']+ df['closevema']

    #df = df.reset_index()
    df = df.sort_values(by=['Symbol','date2'])

    dfsorted = df.groupby('Symbol').tail(1).sort_values(by = ['Total_Score'], ascending=False)
    dfsorted.to_csv('results.csv',index=False)
    
    return dfsorted



st.title('STOCKSY APP | DAILY STOCK PICKS')


#stocks = save_sp500_tickers()
#with st.spinner('Wait for it...'):
df = get_stocks_data()

df= df.unstack().reset_index().pivot_table(index=['Ticker','Date'],columns=['Price'],values=0).reset_index().rename(columns={"Ticker": "Symbol",'Volume': "Vol"})
df = get_results(df)
df = df[['Symbol','Date','slope13_normal','Total_Score']].sort_values(by = ['Total_Score','slope13_normal'], ascending=[False,False])
st.dataframe(df)

reccs_l = [x for x in df.sort_values(by = ['Total_Score','slope13_normal'], ascending=[False,False]).head(10)['Symbol']]
reccs = ','.join(reccs_l)

st.subheader('Top 10 reccs ranked by score and slope')
st.text(reccs)

enter_stocks = st.text_input('Search current rank of stocks', 'NFLX,TSLA')
st.write('Stocks selected are: ', enter_stocks)
stocks_list = enter_stocks.split(",")
st.dataframe(df.loc[df['Symbol'].isin(stocks_list)][['Symbol','Total_Score']])


def get_html(ticker):
	code = """
	<!-- TradingView Widget BEGIN -->
	<div class="tradingview-widget-container">
	  <div id="tradingview_705c5"></div>
	  <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
	  <script type="text/javascript">
	  new TradingView.widget(
	  {{
	  "width": 500,
	  "height": 500,
	  "symbol": "{}",
	  "interval": "D",
	  "timezone": "Etc/UTC",
	  "theme": "dark",
	  "style": "1",
	  "locale": "en",
	  "toolbar_bg": "#f1f3f6",
	  "enable_publishing": false,
	  "allow_symbol_change": true,
	  "container_id": "tradingview_705c5"
	}}
	  );
	  </script>
	</div>
	<!-- TradingView Widget END -->

	""".format(ticker)

	return code


col1, col2, col3 = st.columns(3)


with col1:
	h = get_html(reccs_l[0])
	components.html(h,height=500, width=500)

	h = get_html(reccs_l[4])
	components.html(h,height=500, width=500)

	h = get_html(reccs_l[7])
	components.html(h,height=500, width=500)

with col2:
	h = get_html(reccs_l[1])
	components.html(h,height=500, width=500)

	h = get_html(reccs_l[5])
	components.html(h,height=500, width=500)

	h = get_html(reccs_l[8])
	components.html(h,height=500, width=500)

with col3:
	h = get_html(reccs_l[2])
	components.html(h,height=500, width=500)

	h = get_html(reccs_l[6])
	components.html(h,height=500, width=500)

	h = get_html(reccs_l[9])
	components.html(h,height=500, width=500)
