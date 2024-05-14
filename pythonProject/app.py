from flask import Flask, render_template, request
import pickle
import numpy as np

popular_df = pickle.load(open('popular.pkl','rb'))
pt = pickle.load(open('pt.pkl','rb'))
item_count = pickle.load(open('item_count.pkl','rb'))
menu = pickle.load(open('menu.pkl','rb'))
similarity_scores = pickle.load(open('similarity_scores.pkl','rb'))

app = Flask(__name__, static_url_path='/static')

@app.route('/')
def index():
    return render_template('index.html',
                           item_count=item_count,
                           item_name =list(popular_df['ItemName'].values),
                           description=list(popular_df['Description'].values),
                           price=list(popular_df['Price'].values)
                           )

@app.route('/recommend')
def recommend_ui():
    return render_template('recommend.html')

@app.route('/recommend_menu', methods=["POST"])
def recommend():
    user_input = request.form.get('user_input')
    index = np.where(pt.index == user_input)[0][0]
    similar_items = sorted(list(enumerate(similarity_scores[index])), key=lambda x: x[1], reverse=True)[1:6]

    data = []
    for i in similar_items:
        item = []
        temp_df = menu[menu['ItemName'] == pt.index[i[0]]]
        item.extend(list(temp_df.drop_duplicates('ItemName')['ItemName'].values))
        item.extend(list(temp_df.drop_duplicates('ItemName')['Description'].values))
        item.extend(list(temp_df.drop_duplicates('ItemName')['ImageURL'].values))
        item.extend(list(temp_df.drop_duplicates('ItemName')['Price'].values))

        data.append(item)

    print(data)

    return render_template('recommend.html', data = data)

if __name__ == '__main__':
    app.run(debug = True)