import tensorflow_hub as hub
import numpy as np
from nltk.tokenize import word_tokenize
import pandas as pd
from scipy.spatial import distance
from flask import Flask, request, jsonify
from flask_cors import CORS


app = Flask(__name__)
CORS(app)
clubs_df = pd.read_csv("data_USE.csv")
path= "https://tfhub.dev/google/universal-sentence-encoder/4"
model = hub.load(path)
stop_words = [
    "a", "about", "above", "after", "again", "against", "ain", "all", "am", "an", "and", "any", "are", "aren",
    "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can", "couldn",
    "couldn't", "d", "did", "didn", "didn't", "do", "does", "doesn", "doesn't", "doing", "don", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn", "hadn't", "has", "hasn", "hasn't", "have", "haven", "haven't",
    "having", "he", "her", "here", "hers", "herself", "him", "himself", "his", "how", "i", "if", "in", "into", "is", "isn",
    "isn't", "it", "it's", "its", "itself", "just", "ll", "m", "ma", "me", "mightn", "mightn't", "more", "most", "mustn", "mustn't",
    "my", "myself", "needn", "needn't", "no", "nor", "not", "now", "o", "of", "off", "on", "once", "only", "or", "other", "our",
    "ours", "ourselves", "out", "over", "own", "re", "s", "same", "shan", "shan't", "she", "she's", "should", "shouldn",
    "shouldn't", "so", "some", "such", "t", "than", "that", "that'll", "the", "their", "theirs", "them", "themselves", "then",
    "there", "these", "they", "this", "those", "through", "to", "too", "under", "until", "up", "ve", "very", "was", "wasn",
    "wasn't", "we", "were", "weren", "weren't", "what", "when", "where", "which", "while", "who", "whom", "why", "will", "with",
    "won", "won't", "wouldn", "wouldn't", "y", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves"
]


def similarity(embedding_string, embedded_input):
    stripped_embedding = embedding_string.strip("[]").split()
    float_embedded = np.array([float(i) for i in stripped_embedding])
    return 1-distance.cosine(float_embedded,embedded_input)

@app.route("/clubs", methods=['POST'])
def find_clubs():
    #query = input("Enter a club's name\n")
    #query = ""
    if True:
        test = [
        {
            "name": "a",
            "description_summary": "a",
            "description": "a",
            "contact": "a",
            "meeting": "a"
        },        {
            "name": "a",
            "description_summary": "a",
            "description": "a",
            "contact": "a",
            "meeting": "a"
        },        {
            "name": "a",
            "description_summary": "a",
            "description": "a",
            "contact": "a",
            "meeting": "a"
        }
        ]
        return jsonify(test)
    query = request.json.get("data").get("text")
    print("query: " + query + " : end Query")
    query = word_tokenize(query)
    filtered_query = ' '.join([word for word in query if word.lower() not in stop_words])
    embedded_query = model([filtered_query])[0].numpy()

    clubs_df['similarity'] = clubs_df['description_embeddings'].apply(
        lambda element : similarity(element, embedded_query)
    )
    top5 = clubs_df.sort_values("similarity", ascending=False).head(5)
    # top5_dict = {
    #     "name" : top5["name"].tolist(),
    #     "description_summary" : top5["description_summary"].tolist(),
    #     "description" : top5["description"].tolist(),
    #     "contact" : top5["contact"].tolist(),
    #     "meeting" : top5["meeting"].tolist()
    # }
    # return top5_dict
    top5_list = [
        {
            "name": row["name"],
            "description_summary": row["description_summary"],
            "description": row["description"],
            "contact": row["contact"],
            "meeting": row["meeting"]
        }
        for _, row in top5.iterrows()
    ]

    print("Response data:", top5_list) 
    return jsonify(top5_list)
    #return None

#find_clubs()
if __name__ == '__main__':
    app.run()

