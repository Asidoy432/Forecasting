from flask import Flask, request, jsonify
import joblib, json, numpy as np, pandas as pd

app = Flask(__name__)

# Load all 3 models once at startup
model_daily   = joblib.load('model_daily.pkl')
model_weekly  = joblib.load('model_weekly.pkl')
model_monthly = joblib.load('model_monthly.pkl')

le       = json.load(open('label_encoders.json'))
feat_map = json.load(open('feature_cols.json'))

def encode(col, val):
    return le.get(col, {}).get(val, 0)

@app.route('/')
def home():
    return jsonify({'status': 'ok', 'message': 'Forecasting API running'})

@app.route('/forecast', methods=['POST'])
def forecast():
    try:
        d = request.get_json()

        product  = d.get('product', '')
        category = d.get('category', '')
        branch   = d.get('branch', '')
        period   = d.get('period', 'daily')  # daily / weekly / monthly
        price    = float(d.get('price', 50))
        month    = int(d.get('month', 1))
        quarter  = int(d.get('quarter', 1))
        week     = int(d.get('week', 1))
        txn      = int(d.get('transactions', 10))

        cat_enc    = encode('Category', category)
        prod_enc   = encode('Product_Name', product)
        branch_enc = encode('Store_Branch', branch)

        if period == 'daily':
            row = {
                'Category_Enc': cat_enc, 'Product_Name_Enc': prod_enc,
                'Store_Branch_Enc': branch_enc,
                'DayOfWeek'  : int(d.get('day_of_week', 0)),
                'DayOfMonth' : int(d.get('day_of_month', 1)),
                'MonthNum'   : month, 'WeekOfYear': week,
                'IsWeekend'  : int(d.get('is_weekend', 0)),
                'Quarter'    : quarter, 'Avg_Price': price,
                'Transactions': txn
            }
            model = model_daily

        elif period == 'weekly':
            row = {
                'Category_Enc': cat_enc, 'Product_Name_Enc': prod_enc,
                'Store_Branch_Enc': branch_enc,
                'WeekOfYear': week, 'MonthNum': month,
                'Quarter': quarter, 'Avg_Price': price,
                'Transactions': txn
            }
            model = model_weekly

        else:  # monthly
            row = {
                'Category_Enc': cat_enc, 'Product_Name_Enc': prod_enc,
                'Store_Branch_Enc': branch_enc,
                'MonthNum': month, 'Quarter': quarter,
                'Avg_Price': price, 'Transactions': txn
            }
            model = model_monthly

        features = feat_map[period]
        X   = pd.DataFrame([{f: row[f] for f in features}])
        qty = max(1, int(round(model.predict(X)[0])))

        return jsonify({
            'status'          : 'ok',
            'product'         : product,
            'period'          : period,
            'predicted_qty'   : qty,
            'restock_suggest' : qty + max(1, qty // 5)
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
