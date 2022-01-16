
from flask import Flask, redirect, url_for, request, render_template
from text_summarizer import Summariser

##creating a flask app and naming it "app"
app = Flask('app')


# Model saved with Keras model.save()
MODEL_PATH = '/Text_Summarizer.h5'

# Load your trained model
model = load_model(MODEL_PATH)
model._make_predict_function()          # Necessary
# print('Model loaded. Start serving...')


@app.route('/', methods=['GET'])

def index():
    return render_template("index.html")

# def test():
#     return 'Pinging Model Application!!'

@app.route('/predict', methods=['GET', 'POST'])
def result():
    if request.method == "POST":
        text = request.form["text"]
        if text == "":
            return render_template("home.html")

        else:
            paragraph = Summariser.clean_text(text)
            freq_table = Summariser.create_frequency_table(paragraph)
            sentence = Summariser.sentence_tokenize(paragraph)
            sentence_scores = Summariser.score_sentences(sentence, freq_table)
            threshold = Summariser.find_average_score(sentence_scores)
            summary = Summariser.generate_summary(sentence, sentence_scores, 0.75 * threshold)
            
            return render_template("result.html",text=summary)



if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)