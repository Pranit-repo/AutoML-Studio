from fileinput import filename
import os
import sys
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
import re
from textblob import TextBlob
import distance
from io import StringIO
import json

app = Flask(__name__)
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'upload')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


@app.route('/')
def index():
    return render_template('clean_index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files.get('file')
    if not file:
        return jsonify({'error': 'No file uploaded'})

    df = pd.read_csv(file)
    output = io.StringIO()
    df.to_csv(output, index=False)
    csv_content = output.getvalue()

    return jsonify({
        'file_content': csv_content,
        'preview': df.head(20).to_html(),
        'stats': get_stats(df),
        'columns': df.columns.tolist()
    })


def get_stats(df):
    stats = {}
    for col in df.columns:
        dtype = str(df[col].dtype)
        unique = int(df[col].nunique())  
        missing = int(df[col].isnull().sum())  
        sample = df[col].dropna().sample(min(5, len(df))).values
        
       
        sample = [int(x) if isinstance(x, np.integer) else 
                 float(x) if isinstance(x, np.floating) else 
                 str(x) if isinstance(x, (np.bool_, np.object_)) else x 
                 for x in sample]
        
        stats[col] = {
            'dtype': dtype,
            'unique': unique,
            'missing': missing,
            'sample': sample
        }
    return stats
import io

@app.route('/clean', methods=['POST'])
def clean_data():
    data = request.json
    operations = data.get('operations', [])
    file_content = data.get('file_content')

    if not file_content:
        return jsonify({'error': 'Missing file content'})

    try:
        df = pd.read_csv(io.StringIO(file_content))  

        for op in operations:
            df = apply_operation(df, op)

        output = io.StringIO()
        df.to_csv(output, index=False)
        cleaned_csv = output.getvalue()

        return jsonify({
            'success': True,
            'preview': df.head(20).to_html(),
            'stats': get_stats(df),
            'file_content': cleaned_csv
        })
    except Exception as e:
        return jsonify({'error': str(e)})

def apply_operation(df, operation):
    op_type = operation['type']
    params = operation.get('params', {})
    
    if op_type == 'delete_columns':
        return df.drop(columns=params['columns'])
    
    elif op_type == 'drop_na':
        if params['method'] == 'all':
            return df.dropna(how='all')
        elif params['method'] == 'any':
            return df.dropna(how='any')
        else:  
            return df.dropna(thresh=params['threshold'])
    
    elif op_type == 'impute':
        column = params['column']
        method = params['method']
        
        if method == 'mean':
            df[column] = df[column].fillna(df[column].mean())
        elif method == 'median':
            df[column] = df[column].fillna(df[column].median())
        elif method == 'mode':
            df[column] = df[column].fillna(df[column].mode()[0])
        elif method == 'knn':
            imputer = KNNImputer(n_neighbors=params.get('neighbors', 5))
            df[column] = imputer.fit_transform(df[[column]]).flatten()
        elif method == 'mice':
            
            from sklearn.experimental import enable_iterative_imputer
            from sklearn.impute import IterativeImputer
            imputer = IterativeImputer(max_iter=10, random_state=0)
            df[column] = imputer.fit_transform(df[[column]]).flatten()
        
        return df
    
    elif op_type == 'outlier':
        column = params['column']
        method = params['method']
        
        if method == 'zscore':
            z_scores = (df[column] - df[column].mean()) / df[column].std()
            if params['action'] == 'remove':
                df = df[abs(z_scores) < params['threshold']]
            else:  
                upper = df[column].mean() + params['threshold'] * df[column].std()
                lower = df[column].mean() - params['threshold'] * df[column].std()
                df[column] = df[column].clip(lower, upper)
        
        elif method == 'iqr':
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - params['factor'] * IQR
            upper = Q3 + params['factor'] * IQR
            
            if params['action'] == 'remove':
                df = df[(df[column] >= lower) & (df[column] <= upper)]
            else:  # cap
                df[column] = df[column].clip(lower, upper)
        
        elif method == 'isolation_forest':
            clf = IsolationForest(contamination=params.get('contamination', 0.1))
            preds = clf.fit_predict(df[[column]])
            if params['action'] == 'remove':
                df = df[preds == 1]
            else:  
                df.loc[preds == -1, column] = np.nan
        
        return df
    
    elif op_type == 'normalize':
        column = params['column']
        method = params['method']
        
        if method == 'minmax':
            scaler = MinMaxScaler()
            df[column] = scaler.fit_transform(df[[column]]).flatten()
        elif method == 'zscore':
            scaler = StandardScaler()
            df[column] = scaler.fit_transform(df[[column]]).flatten()
        
        return df
    
    elif op_type == 'binning':
        column = params['column']
        method = params['method']
        bins = params['bins']
        
        if method == 'equal_width':
            df[column+'_binned'] = pd.cut(df[column], bins=bins)
        elif method == 'equal_freq':
            df[column+'_binned'] = pd.qcut(df[column], q=bins)
        
        return df
    
    elif op_type == 'text_clean':
        column = params['column']
        
        if params['action'] == 'lowercase':
            df[column] = df[column].str.lower()
        elif params['action'] == 'remove_punct':
            df[column] = df[column].str.replace(r'[^\w\s]', '')
        elif params['action'] == 'remove_stopwords':
            from nltk.corpus import stopwords
            stop_words = set(stopwords.words('english'))
            df[column] = df[column].apply(lambda x: ' '.join([word for word in str(x).split() if word not in stop_words]))
        elif params['action'] == 'correct_spelling':
            df[column] = df[column].apply(lambda x: str(TextBlob(str(x)).correct()))
        
        return df
    
    elif op_type == 'deduplicate':
        if params['method'] == 'exact':
            return df.drop_duplicates(subset=params.get('columns', None))
        elif params['method'] == 'fuzzy':
            
            from fuzzywuzzy import fuzz
            threshold = params.get('threshold', 80)
            column = params['column']
            
            duplicates = set()
            for i in range(len(df)):
                if i in duplicates:
                    continue
                for j in range(i+1, len(df)):
                    if fuzz.ratio(str(df.at[i, column]), str(df.at[j, column])) > threshold:
                        duplicates.add(j)
            
            return df.drop(index=list(duplicates))
    
    elif op_type == 'type_conversion':
        column = params['column']
        new_type = params['type']
        
        if new_type == 'numeric':
            df[column] = pd.to_numeric(df[column], errors='coerce')
        elif new_type == 'datetime':
            df[column] = pd.to_datetime(df[column], errors='coerce')
        elif new_type == 'category':
            df[column] = df[column].astype('category')
        elif new_type == 'string':
            df[column] = df[column].astype(str)
        
        return df
    
    elif op_type == 'regex_extract':
        column = params['column']
        pattern = params['pattern']
        new_column = params.get('new_column', f"{column}_extracted")
        
        df[new_column] = df[column].str.extract(pattern)
        return df
    
    return df

if __name__ == '__main__':
    app.run(debug=True, port=5003)