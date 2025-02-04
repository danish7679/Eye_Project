from flask import Flask, render_template,request,session,flash
import sqlite3 as sql
import os
from csv import writer
# %%
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import subprocess

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route('/gohome')
def homepage():
    return render_template('index.html')

@app.route('/enternew')
def new_user():
   return render_template('signup.html')

@app.route('/addrec',methods = ['POST', 'GET'])
def addrec():
    if request.method == 'POST':
        try:
            nm = request.form['Name']
            phonno = request.form['MobileNumber']
            email = request.form['email']
            unm = request.form['Username']
            passwd = request.form['password']

            with sql.connect("multisearch1.db") as con:
                cur = con.cursor()
                cur.execute("INSERT INTO muser(name,phono,email,username,password)VALUES(?, ?, ?, ?,?)", (nm, phonno, email, unm, passwd))
                con.commit()
                msg = "Record successfully added"
        except:
            con.rollback()
            msg = "error in insert operation"

        finally:
            return render_template("login.html")
            con.close()


@app.route('/userlogin')
def login_user():
    return render_template('login.html')

@app.route('/predict')
def info_user():
    return render_template('info.html')

@app.route('/logindetails',methods = ['POST', 'GET'])
def logindetails():
    if request.method=='POST':
            usrname=request.form['username']
            passwd = request.form['password']

            with sql.connect("multisearch1.db") as con:
                cur = con.cursor()
                cur.execute("SELECT username,password FROM muser where username=? ",(usrname,))
                account = cur.fetchall()

                for row in account:
                    database_user = row[0]
                    database_password = row[1]
                    if database_user == usrname and database_password==passwd:
                        session['logged_in'] = True
                        subprocess.call(" streamlit run streamlit_test.py 1", shell=True)
                        return render_template('search.html')
                    else:
                        flash("Invalid user credentials")
                        return render_template('login.html')

@app.route('/reply',methods = ['POST', 'GET'])
def user_reply():
    if request.method=='POST':
        ques=request.form['searchword']
        print('ques', ques)

        X_test1 = ques.lower()
        X_test1 = [X_test1]
        print('length of test1',len(ques))
        if len(ques) == 0:
            response = 'Enter Proper Text'
            response1 = '0'
            return render_template('resultpred.html', prediction=response, prediction1= response1)
        
        print('X_test1', X_test1)
        raw_mail_data=pd.read_csv('mail_data1.csv')
        print(raw_mail_data)

        # %%
        #Replace the null values with null string
        mail_data=raw_mail_data.where((pd.notnull(raw_mail_data)),'')

        # %%
        #print 5 rows
        mail_data.head()

        # %%
        #checking no of rows and coloums
        mail_data.shape

        # %%
        #Label encoding
        #label spam mail as 0; ham as 1;
        mail_data.loc[mail_data['Category'] == 'spam', 'Category',] = 0
        mail_data.loc[mail_data['Category'] == 'ham','Category',] = 1

        # %%
        #seperate data has text and labels
        X=mail_data['Message']
        Y=mail_data['Category']

        # %%
        print(X)

        # %%
        print(Y)

        # %%
        #spliting data into train and test
        X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size=0.2, random_state=3)

        # %%
        print(X.shape)
        print(X_train.shape)
        print(X_test.shape)

        # %%
        #transform text data to feature vectors that can be used as input to logistic regression model
        feature_extraction = TfidfVectorizer(min_df=1, stop_words="english", lowercase="True")
        X_train_features = feature_extraction.fit_transform(X_train)
        X_test_features = feature_extraction.transform(X_test)
        #X_test1 = ['Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&Cs apply 08452810075over18s']
        X_test_features1 = feature_extraction.transform(X_test1)
        #convert Y_train and Y_test values as integers
        Y_train = Y_train.astype('int')
        Y_test = Y_test.astype('int')

        # %%
        print(X_train)

        # %%
        print(X_train_features)

        # %%
        # Training Model
        #Logistic Regression
        model = LogisticRegression()

        # %%
        # Training the logistic Regression MODEL with training data
        model.fit(X_train_features, Y_train)


        # %%
        #Predict on training data
        prediction_on_training_data = model.predict(X_train_features)
        accuracy_on_training_data = accuracy_score(Y_train, prediction_on_training_data)

        # %%
        print('Accuracy on training data :',accuracy_on_training_data)


        # %%
        #prediction on test data
        prediction_on_test_data = model.predict(X_test_features)

        prediction_on_test_data1 = model.predict(X_test_features1)
        print('prediction_on_test_data1',prediction_on_test_data1)
        accuracy_on_test_data = accuracy_score(Y_test, prediction_on_test_data)

        # %%
        print('Accuracy on testing data :',accuracy_on_test_data)
        response1 = int(accuracy_on_test_data*100)
        
        pred1 = prediction_on_test_data1
        print('prediction', pred1[0])
        if pred1[0] == 0:
            print('spam')
            response = 'spam'
        else:
            print('ham')
            response = 'ham'
        




        

       

      

        #response= 'news updates'




        #return render_template('search.html')
        return render_template('resultpred.html', prediction=response, prediction1= response1)

@app.route('/predict',methods = ['POST', 'GET'])

def predcrop():
    if request.method == 'POST':
        comment = request.form['comment']
        comment1 = request.form['comment1']
        comment2 = request.form['comment2']
        data = comment
        data1 = comment1
        data2 = comment2
        # type(data2)
        print(data)
        print(data1)
        print(data2)
        List = [data, data1, data2]
        with open('events.csv', 'a') as f_object:
            writer_object = writer(f_object)
            writer_object.writerow(List)
            f_object.close()
            response = 'Thank you for Feedback'
    return render_template('resultpred1.html', prediction=response)





@app.route("/logout")
def logout():
    session['logged_in'] = False
    return render_template('login.html')

if __name__ == '__main__':
    app.secret_key = os.urandom(12)
    app.run(debug=True)




