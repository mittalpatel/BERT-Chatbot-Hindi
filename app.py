# ==============================================================================
# title              : app.py
# description        : This is the flask app for Bert closed domain chatbot which accepts the user request and response back with the answer
# author             : Pragnakalp Techlabs
# email              : letstalk@pragnakalp.com
# website            : https://www.pragnakalp.com
# python_version     : 3.6.x +
# ==============================================================================

# Import required libraries
from flask import Flask, render_template, request
from flask_cors import CORS
import email
import csv
import datetime
import smtplib
import ssl
import socket
from email.mime.text import MIMEText
from bert import QA

timestamp = datetime.datetime.now()
date = timestamp.strftime('%d-%m-%Y')
time = timestamp.strftime('%I:%M:%S')
IP = ''

app = Flask(__name__)
CORS(app)

# Provide the fine_tuned model path in QA Class
model_hi = QA("hindi_model_bin")

# This is used to show the home page
@app.route("/")
def home():
    return render_template("home.html")

# This is used to give response 
@app.route("/predict")
def get_bot_response():   
    IP = request.remote_addr
    q = request.args.get('msg')
    bert_bot_log = []
    bert_bot_log.append(q)
    bert_bot_log.append(date)
    bert_bot_log.append(time)
    bert_bot_log.append(IP)
    
    # You can provide your own paragraph from here
    hindi_para = "भारत (आधिकारिक नाम: भारत गणराज्य, अंग्रेज़ी: Republic of India) दक्षिण एशिया में स्थित भारतीय उपमहाद्वीप का सबसे बड़ा देश है। पूर्ण रूप से उत्तरी गोलार्ध में स्थित भारत, भौगोलिक दृष्टि से विश्व में छठा सबसे बड़ा और जनसंख्या के दृष्टिकोण से दूसरा सबसे बड़ा देश है। भारत के पश्चिम में पाकिस्तान, उत्तर-पूर्व में चीन, नेपाल और भूटान, पूर्व में बांग्लादेश और म्यान्मार स्थित हैं। हिन्द महासागर में इसके दक्षिण पश्चिम में मालदीव, दक्षिण में श्रीलंका और दक्षिण-पूर्व में इंडोनेशिया से भारत की सामुद्रिक सीमा लगती है। इसके उत्तर की भौतिक सीमा हिमालय पर्वत से और दक्षिण में हिन्द महासागर से लगी हुई है। पूर्व में बंगाल की खाड़ी है तथा पश्चिम में अरब सागर हैं। प्राचीन सिन्धु घाटी सभ्यता, व्यापार मार्गों और बड़े-बड़े साम्राज्यों का विकास-स्थान रहे भारतीय उपमहाद्वीप को इसके सांस्कृतिक और आर्थिक सफलता के लंबे इतिहास के लिये जाना जाता रहा है। चार प्रमुख संप्रदायों: हिंदू, बौद्ध, जैन और सिख धर्मों का यहां उदय हुआ।"

    # This function creates a log file which contain the question, answer, date, time, IP addr of the user
    def bert_log_fn(answer_err):
        bert_bot_log.append(answer_err)
        with open('bert_bot_log.csv', 'a' , encoding='utf-8') as logs:
            write = csv.writer(logs)
            write.writerow(bert_bot_log)
        logs.close()

    # This block calls the prediction function and return the response
    try:        
        out = model_hi.predict(hindi_para, q)
        confidence = out["confidence"]
        confidence_score = round(confidence*100)
        if confidence_score > 10:
            bert_log_fn(out["answer"])
            return out["answer"]
        else:
            bert_log_fn("Sorry I don't know the answer, please try some different question.")
            return "Sorry I don't know the answer, please try some different question."         
    except Exception as e:
        bert_log_fn("Sorry, Server doesn't respond..!!")
        print("Exception Message ==> ",e)
        return "Sorry, Server doesn't respond..!!"

# You can change the Flask app port number from here.
if __name__ == "__main__":
    app.run(port='3000')
