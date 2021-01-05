from flask import Flask,jsonify, request

app = Flask(__name__)

@app.route('/engine', methods=["GET", "POST"])
def engine():
    if request.method == 'GET':
        algo = request.args.get('algo')
        ptype = request.args.get('ptype')
        tick= request.args.get('tick')
        daysx = request.args.get('daysx')
        if algo == "Delta":
            algo="adadelta"
        elif algo == "Meta":
            algo="adam"
        elif algo == "Gradient":
            algo = "adagrad"
        #importing the packages
        #part 1
        daysx=int(daysx)
        import datetime as dt
        import urllib.request, json
        import pandas as pd #3
        import numpy as np #3
#        import matplotlib.pyplot as plt
#        from matplotlib.pylab import rcParams
        from sklearn.preprocessing import MinMaxScaler
        #used for setting the output figure size
#        rcParams['figure.figsize'] = 20,10
        #to normalize the given input data
        scaler = MinMaxScaler(feature_range=(0, 1))
        #to read input data set (place the file name inside  ' ') as shown below
        ticker = tick
        
        api_key = '3T9YAWICQG9J42JM'
        
        url_string = "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=%s&outputsize=compact&apikey=%s"%(ticker,api_key)
        
        todataframe=pd.DataFrame()
        
        with urllib.request.urlopen(url_string) as url:
                    datax = json.loads(url.read().decode())
        
                    datax = datax['Time Series (Daily)']
                    df = pd.DataFrame(columns=['Date','Open','High','Low','Close','Volume'])
                    for k,v in datax.items():
                        date = dt.datetime.strptime(k, '%Y-%m-%d')
                        data_row = [date.date(),float(v['3. low']),float(v['2. high']),float(v['4. close']),float(v['1. open']),float(v['5. volume'])]
        #                print(data_row)
                        df.loc[-1,:] = data_row
                        df.index = df.index + 1
        
        todataframe=df
        #todataframe.head() 
        #print(todataframe)
        
        #part 2
        
        #todataframe['Date'] = pd.to_datetime(todataframe.Date,format='%Y-%m-%d')
        #todataframe.index = todataframe['Date']
        #plt.figure(figsize=(16,8))
        #plt.plot(todataframe['Close'], label='Closing Price')
        
        #part 3
        
        from sklearn.preprocessing import MinMaxScaler
        from tensorflow.python.keras.layers import Dense, Dropout, LSTM
        from tensorflow.python.keras import Sequential
        #dataframe creation
        seriesdata = todataframe.sort_index(ascending=True, axis=0)
        new_seriesdata = pd.DataFrame(index=range(0,len(todataframe)),columns=['Date',ptype])
        length_of_data=len(seriesdata)
        for i in range(0,length_of_data):
            new_seriesdata['Date'][i] = seriesdata['Date'][i]
            new_seriesdata[ptype][i] = seriesdata[ptype][i]
        #setting the index again
        new_seriesdata.index = new_seriesdata.Date
        new_seriesdata.drop('Date', axis=1, inplace=True)
        #creating train and test sets this comprises the entire data’s present in the dataset
        myseriesdataset = new_seriesdata.values
        totrain = myseriesdataset[0:new_seriesdata.size,:]
        tovalid = myseriesdataset[new_seriesdata.size:,:]
        
        #print(len(totrain))
        #print(len(tovalid))
        
        
        #part 4
        
        scalerdata = MinMaxScaler(feature_range=(0, 1))
        scale_data = scalerdata.fit_transform(myseriesdataset)
        x_totrain, y_totrain = [], []
        length_of_totrain=len(totrain)
        for i in range(60,length_of_totrain):
            x_totrain.append(scale_data[i-60:i,0])
            y_totrain.append(scale_data[i,0])
        x_totrain, y_totrain = np.array(x_totrain), np.array(y_totrain)
        x_totrain = np.reshape(x_totrain, (x_totrain.shape[0],x_totrain.shape[1],1))
        #LSTM neural network
        lstm_model = Sequential()
        lstm_model.add(LSTM(units=50, return_sequences=True, input_shape=(x_totrain.shape[1],1)))
        lstm_model.add(LSTM(units=50))
        lstm_model.add(Dense(1))
        lstm_model.compile(loss='mean_squared_error', optimizer=algo)
        lstm_model.fit(x_totrain, y_totrain, epochs=3, batch_size=1, verbose=2)
        #predicting next data stock price
        myinputs = new_seriesdata[len(new_seriesdata) - (len(tovalid)+daysx) - 60:].values
        myinputs = myinputs.reshape(-1,1)
        myinputs  = scalerdata.transform(myinputs)
        tostore_test_result = []
        for i in range(60,myinputs.shape[0]):
            tostore_test_result.append(myinputs[i-60:i,0])
        tostore_test_result = np.array(tostore_test_result)
        tostore_test_result = np.reshape(tostore_test_result,(tostore_test_result.shape[0],tostore_test_result.shape[1],1))
        myclosing_priceresult = lstm_model.predict(tostore_test_result)
        myclosing_priceresult = scalerdata.inverse_transform(myclosing_priceresult)
        
        #part 5
        
        #print(len(tostore_test_result));
#        print(myclosing_priceresult);
        xm=myclosing_priceresult.tolist()
        
        
        return jsonify(xm)
    
        

