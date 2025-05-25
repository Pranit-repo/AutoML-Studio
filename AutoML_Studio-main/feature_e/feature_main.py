import sys
from flask import Flask, render_template, request, jsonify, send_from_directory
import pandas as pd
import numpy as np
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    OneHotEncoder, LabelEncoder
)
from sklearn.feature_selection import (
    SelectKBest, chi2, f_classif, mutual_info_classif,
    RFE
)
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer, KNNImputer
import os
import json
from datetime import datetime
from scipy import stats
import hashlib

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 


os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    return render_template('feature_index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        filename = hashlib.md5((file.filename + str(datetime.now())).encode()).hexdigest() + '.csv'
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            df = pd.read_csv(filepath)
            return jsonify({
                'filename': filename,
                'columns': list(df.columns),
                'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
                'preview': df.head(20).to_dict(orient='records'),
                'missing': df.isnull().sum().to_dict(),
                'stats': df.describe().to_dict()
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 400

@app.route('/process', methods=['POST'])
def process_data():
    data = request.get_json()
    filename = data['filename']
    operations = data['operations']
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    df = pd.read_csv(filepath)
    
    results = {}
    
    for op in operations:
        try:
            if op['type'] == 'missing':
                df = handle_missing(df, op)
            elif op['type'] == 'encoding':
                df = handle_encoding(df, op)
            elif op['type'] == 'scaling':
                df = handle_scaling(df, op)
            elif op['type'] == 'transformation':
                df = handle_transformation(df, op)
            elif op['type'] == 'feature_selection':
                df, results[op['type']] = handle_feature_selection(df, op)
            elif op['type'] == 'outliers':
                df = handle_outliers(df, op)
            elif op['type'] == 'dimensionality':
                df, results[op['type']] = handle_dimensionality(df, op)
            elif op['type'] == 'feature_creation':
                df = handle_feature_creation(df, op)
            elif op['type'] == 'binning':
                df = handle_binning(df, op)
        except Exception as e:
            return jsonify({'error': str(e), 'operation': op}), 400
    
    processed_filename = 'processed_' + filename
    processed_filepath = os.path.join(app.config['UPLOAD_FOLDER'], processed_filename)
    df.to_csv(processed_filepath, index=False)
    
    return jsonify({
        'processed_filename': processed_filename,
        'preview': df.head(20).to_dict(orient='records'),
        'results': results
    })

def handle_missing(df, op):
    strategy = op['strategy']
    columns = op['columns']
    
    for col in columns:
        if strategy == 'mean':
            df[col].fillna(df[col].mean(), inplace=True)
        elif strategy == 'median':
            df[col].fillna(df[col].median(), inplace=True)
        elif strategy == 'mode':
            df[col].fillna(df[col].mode()[0], inplace=True)
        elif strategy == 'constant':
            df[col].fillna(op.get('value', 0), inplace=True)
        elif strategy == 'knn':
            imputer = KNNImputer(n_neighbors=op.get('n_neighbors', 5))
            df[col] = imputer.fit_transform(df[[col]])
        elif strategy == 'drop':
            df.dropna(subset=[col], inplace=True)
        elif strategy == 'flag':
            df[f'{col}_missing'] = df[col].isnull().astype(int)
            df[col].fillna(df[col].median(), inplace=True)
    
    return df

def handle_encoding(df, op):
    method = op['method']
    columns = op['columns']
    
    for col in columns:
        if method == 'onehot':
            encoder = OneHotEncoder(sparse=False, drop=op.get('drop', 'first'))
            encoded = encoder.fit_transform(df[[col]])
            encoded_df = pd.DataFrame(encoded, columns=[f"{col}_{cat}" for cat in encoder.categories_[0]])
            df = pd.concat([df.drop(col, axis=1), encoded_df], axis=1)
        elif method == 'label':
            encoder = LabelEncoder()
            df[col] = encoder.fit_transform(df[col])
        elif method == 'target':
            target = op['target']
            means = df.groupby(col)[target].mean()
            df[col] = df[col].map(means)
        elif method == 'frequency':
            freq = df[col].value_counts(normalize=True)
            df[col] = df[col].map(freq)
        elif method == 'binary':
            df[col] = df[col].astype('category').cat.codes
    
    return df

def handle_scaling(df, op):
    method = op['method']
    columns = op['columns']
    
    for col in columns:
        if method == 'standard':
            scaler = StandardScaler()
            df[col] = scaler.fit_transform(df[[col]])
        elif method == 'minmax':
            scaler = MinMaxScaler(feature_range=tuple(op.get('range', [0, 1])))
            df[col] = scaler.fit_transform(df[[col]])
        elif method == 'robust':
            scaler = RobustScaler()
            df[col] = scaler.fit_transform(df[[col]])
    
    return df

def handle_transformation(df, op):
    method = op['method']
    columns = op['columns']
    
    for col in columns:
        if method == 'log':
            df[col] = np.log1p(df[col])
        elif method == 'sqrt':
            df[col] = np.sqrt(df[col])
        elif method == 'boxcox':
            df[col], _ = stats.boxcox(df[col] + 1)  # +1 to handle zeros
        elif method == 'yeojohnson':
            df[col], _ = stats.yeojohnson(df[col])
    
    return df

def handle_feature_selection(df, op):
    method = op['method']
    target = op.get('target')
    k = op.get('k', 5)
    
    results = {}
    
    if method == 'variance':
        from sklearn.feature_selection import VarianceThreshold
        selector = VarianceThreshold(threshold=op.get('threshold', 0))
        selector.fit(df.select_dtypes(include=['number']))
        selected = df.columns[selector.get_support()]
        results['selected_features'] = list(selected)
        return df[selected], results
    
    elif method == 'correlation':
        corr_matrix = df.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > op.get('threshold', 0.8))]
        results['dropped_features'] = to_drop
        return df.drop(to_drop, axis=1), results
    
    elif method == 'kbest':
        if not target:
            raise ValueError("Target column required for SelectKBest")
        
        score_func = {
            'chi2': chi2,
            'f_classif': f_classif,
            'mutual_info': mutual_info_classif
        }[op.get('score_func', 'f_classif')]
        
        selector = SelectKBest(score_func=score_func, k=k)
        X = df.drop(target, axis=1).select_dtypes(include=['number'])
        y = df[target]
        selector.fit(X, y)
        selected = X.columns[selector.get_support()]
        results['selected_features'] = list(selected)
        results['scores'] = dict(zip(X.columns, selector.scores_))
        return pd.concat([df[selected], df[target]], axis=1), results
    
    elif method == 'rfe':
        if not target:
            raise ValueError("Target column required for RFE")
        
        estimator = LogisticRegression(max_iter=1000)
        selector = RFE(estimator, n_features_to_select=k, step=1)
        X = df.drop(target, axis=1).select_dtypes(include=['number'])
        y = df[target]
        selector.fit(X, y)
        selected = X.columns[selector.get_support()]
        results['selected_features'] = list(selected)
        results['ranking'] = dict(zip(X.columns, selector.ranking_))
        return pd.concat([df[selected], df[target]], axis=1), results
    
    return df, results

def handle_outliers(df, op):
    method = op['method']
    columns = op['columns']
    
    for col in columns:
        if method == 'clip':
            lower = op.get('lower', df[col].quantile(0.05))
            upper = op.get('upper', df[col].quantile(0.95))
            df[col] = df[col].clip(lower, upper)
        elif method == 'remove':
            lower = op.get('lower', df[col].quantile(0.05))
            upper = op.get('upper', df[col].quantile(0.95))
            df = df[(df[col] >= lower) & (df[col] <= upper)]
        elif method == 'transform':
            df[col] = np.log1p(df[col])
    
    return df

def handle_dimensionality(df, op):
    method = op['method']
    n_components = op.get('n_components', 2)
    target = op.get('target')
    
    results = {}
    
    if method == 'pca':
        pca = PCA(n_components=n_components)
        numeric_cols = df.select_dtypes(include=['number']).columns
        if target and target in numeric_cols:
            numeric_cols = numeric_cols.drop(target)
        
        components = pca.fit_transform(df[numeric_cols])
        component_names = [f'PC{i+1}' for i in range(n_components)]
        components_df = pd.DataFrame(components, columns=component_names)
        
        results['explained_variance'] = dict(zip(
            component_names,
            pca.explained_variance_ratio_
        ))
        
        if target:
            return pd.concat([components_df, df[target]], axis=1), results
        return components_df, results
    
    return df, results

def handle_feature_creation(df, op):
    operation = op['operation']
    new_feature = op['new_feature']
    
    if operation == 'interaction':
        col1, col2 = op['columns']
        df[new_feature] = df[col1] * df[col2]
    elif operation == 'polynomial':
        col = op['column']
        degree = op.get('degree', 2)
        df[new_feature] = df[col] ** degree
    elif operation == 'aggregation':
        group_col = op['group_column']
        agg_col = op['aggregation_column']
        agg_func = op['aggregation_function']
        
        if agg_func == 'mean':
            agg_values = df.groupby(group_col)[agg_col].mean()
        elif agg_func == 'sum':
            agg_values = df.groupby(group_col)[agg_col].sum()
        elif agg_func == 'max':
            agg_values = df.groupby(group_col)[agg_col].max()
        elif agg_func == 'min':
            agg_values = df.groupby(group_col)[agg_col].min()
        
        df[new_feature] = df[group_col].map(agg_values)
    elif operation == 'datetime':
        col = op['column']
        part = op['part']
        
        df[col] = pd.to_datetime(df[col])
        if part == 'year':
            df[new_feature] = df[col].dt.year
        elif part == 'month':
            df[new_feature] = df[col].dt.month
        elif part == 'day':
            df[new_feature] = df[col].dt.day
        elif part == 'hour':
            df[new_feature] = df[col].dt.hour
        elif part == 'weekday':
            df[new_feature] = df[col].dt.weekday
        elif part == 'quarter':
            df[new_feature] = df[col].dt.quarter
    
    return df

def handle_binning(df, op):
    method = op['method']
    column = op['column']
    new_feature = op.get('new_feature', f'{column}_binned')
    bins = op.get('bins', 5)
    labels = op.get('labels')
    
    if method == 'equal_width':
        df[new_feature] = pd.cut(df[column], bins=bins, labels=labels)
    elif method == 'equal_freq':
        df[new_feature] = pd.qcut(df[column], q=bins, labels=labels)
    elif method == 'custom':
        bin_edges = op['bin_edges']
        df[new_feature] = pd.cut(df[column], bins=bin_edges, labels=labels)
    
    return df

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True, port=5004)