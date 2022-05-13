from flask import Flask, render_template, request, url_for
import numpy as np
import pickle

model_lg = pickle.load(open("model/model_logreg_FP2.pkl", "rb"))
model_svm = pickle.load(open("model/model_svm_FP2.pkl", "rb"))

app = Flask(__name__, template_folder="templates")


@app.route('/')
def main():
    return render_template('main.html')

# Redirecting the API to predict the result


@app.route("/predict", methods=['POST'])
def predict():
    """
    For Rendering result on HTML GUI
    """
    int_features = [x for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction_lg = model_lg.predict(final_features)
    prediction_svm = model_svm.predict(final_features)

    output_lg = round(prediction_lg[0], 2)
    output_svm = round(prediction_svm[0], 2)
    output_text_lg = ''
    output_text_svm = ''

    if (output_lg == 0 & output_svm == 0):
        output_text_lg = 'Besok tidak hujan'
        output_text_svm = 'Besok tidak hujan'
    elif (output_lg == 0 & output_svm == 1):
        output_text_lg = 'Besok tidak hujan'
        output_text_svm = 'Besok hujan'
    elif (output_lg == 1 & output_svm == 0):
        output_text_lg = 'Besok hujan'
        output_text_svm = 'Besok tidak hujan'
    elif (output_lg == 1 & output_svm == 1):
        output_text_lg = 'Besok hujan'
        output_text_svm = 'Besok hujan'

    return render_template("main.html", prediction_text_lg="Prediksi cuaca besok berdasarkan logistic regression : {}".format(output_text_lg), prediction_text_svm="\n\nPrediksi cuaca besok berdasarkan SVM : {}".format(output_text_svm))


if __name__ == '__main__':
    app.run(debug=True)
