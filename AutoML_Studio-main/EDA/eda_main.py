import sys
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import io
import base64
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import seaborn as sns
from scipy import stats
import json
from statsmodels.stats.outliers_influence import variance_inflation_factor
from wordcloud import WordCloud
import networkx as nx
from mlxtend.frequent_patterns import apriori, association_rules

app = Flask(__name__)


global_df = None

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('eda_index.html')

@app.route('/upload', methods=['POST'])
def upload():
    global global_df
    file = request.files['file']
    if file.filename.endswith('.csv'):
        global_df = pd.read_csv(file)
    elif file.filename.endswith(('.xls', '.xlsx')):
        global_df = pd.read_excel(file)
    else:
        return jsonify({'error': 'Unsupported file format'})
    
    
    info = {
        'shape': global_df.shape,
        'columns': list(global_df.columns),
        'dtypes': global_df.dtypes.astype(str).to_dict(),
        'null_counts': global_df.isnull().sum().to_dict()
    }
    return jsonify(info)

@app.route('/descriptive_stats', methods=['POST'])
def descriptive_stats():
    global global_df
    if global_df is None:
        return jsonify({'error': 'No data uploaded'})
    
    column = request.json.get('column')
    if column not in global_df.columns:
        return jsonify({'error': 'Column not found'})
    
    stats = {
        'mean': global_df[column].mean(),
        'median': global_df[column].median(),
        'mode': global_df[column].mode().tolist(),
        'std': global_df[column].std(),
        'min': global_df[column].min(),
        'max': global_df[column].max(),
        'q1': global_df[column].quantile(0.25),
        'q3': global_df[column].quantile(0.75),
        'skewness': global_df[column].skew(),
        'kurtosis': global_df[column].kurtosis()
    }
    return jsonify(stats)

@app.route('/plot', methods=['POST'])
def plot():
    global global_df
    if global_df is None:
        return jsonify({'error': 'No data uploaded'})
    
    plot_type = request.json.get('plot_type')
    x_col = request.json.get('x_col')
    y_col = request.json.get('y_col')
    hue_col = request.json.get('hue_col')
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    try:
        if plot_type == 'histogram':
            sns.histplot(data=global_df, x=x_col, kde=True, ax=ax)
        elif plot_type == 'boxplot':
            sns.boxplot(data=global_df, x=x_col, y=y_col if y_col else None, ax=ax)
        elif plot_type == 'scatter':
            sns.scatterplot(data=global_df, x=x_col, y=y_col, hue=hue_col if hue_col else None, ax=ax)
        elif plot_type == 'heatmap':
            corr = global_df.select_dtypes(include=['number']).corr()
            sns.heatmap(corr, annot=True, ax=ax)
        elif plot_type == 'violin':
            sns.violinplot(data=global_df, x=x_col, y=y_col if y_col else None, ax=ax)
        elif plot_type == 'pairplot':
            fig = sns.pairplot(global_df.select_dtypes(include=['number'])).fig
        elif plot_type == 'qq':
            stats.probplot(global_df[x_col].dropna(), plot=ax)
        elif plot_type == 'bar':
            sns.barplot(data=global_df, x=x_col, y=y_col if y_col else None, hue=hue_col if hue_col else None, ax=ax)
        elif plot_type == 'line':
            sns.lineplot(data=global_df, x=x_col, y=y_col, hue=hue_col if hue_col else None, ax=ax)
        elif plot_type == 'kde':
            sns.kdeplot(data=global_df, x=x_col, hue=hue_col if hue_col else None, ax=ax)
        elif plot_type == 'pie':
            if y_col:
                ax.pie(global_df[y_col], labels=global_df[x_col], autopct='%1.1f%%')
            else:
                counts = global_df[x_col].value_counts()
                ax.pie(counts, labels=counts.index, autopct='%1.1f%%')
        elif plot_type == 'wordcloud':
            text = ' '.join(global_df[x_col].astype(str))
            wordcloud = WordCloud(width=800, height=400).generate(text)
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
        elif plot_type == 'acf':
            plot_acf(global_df[x_col].dropna(), ax=ax)
        elif plot_type == 'pacf':
            plot_pacf(global_df[x_col].dropna(), ax=ax)
        elif plot_type == 'seasonal_decompose':
            result = seasonal_decompose(global_df.set_index(x_col)[y_col], model='additive', period=12)
            result.plot().set_size_inches(10, 8)
            fig = plt.gcf()
    except Exception as e:
        return jsonify({'error': str(e)})
    
  
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    plt.close(fig)
    
    
    plot_url = base64.b64encode(output.getvalue()).decode('utf8')
    return jsonify({'image': f'data:image/png;base64,{plot_url}'})

@app.route('/outliers', methods=['POST'])
def detect_outliers():
    global global_df
    if global_df is None:
        return jsonify({'error': 'No data uploaded'})
    
    column = request.json.get('column')
    method = request.json.get('method')
    
    if column not in global_df.columns:
        return jsonify({'error': 'Column not found'})
    
    data = global_df[column].dropna()
    
    if method == 'zscore':
        threshold = request.json.get('threshold', 3)
        z_scores = np.abs(stats.zscore(data))
        outliers = data[z_scores > threshold]
    elif method == 'iqr':
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = data[(data < lower_bound) | (data > upper_bound)]
    elif method == 'isolation_forest':
        clf = IsolationForest(contamination=0.05)
        preds = clf.fit_predict(data.values.reshape(-1, 1))
        outliers = data[preds == -1]
    elif method == 'dbscan':
        eps = request.json.get('eps', 0.5)
        min_samples = request.json.get('min_samples', 5)
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(data.values.reshape(-1, 1))
        outliers = data[clustering.labels_ == -1]
    
    return jsonify({
        'outliers': outliers.tolist(),
        'count': len(outliers),
        'percentage': len(outliers) / len(data) * 100
    })

@app.route('/correlation', methods=['POST'])
def calculate_correlation():
    global global_df
    if global_df is None:
        return jsonify({'error': 'No data uploaded'})
    
    method = request.json.get('method')
    col1 = request.json.get('col1')
    col2 = request.json.get('col2')
    
    if col1 not in global_df.columns or col2 not in global_df.columns:
        return jsonify({'error': 'Column not found'})
    
    data1 = global_df[col1]
    data2 = global_df[col2]
    
   
    valid_idx = data1.notna() & data2.notna()
    data1 = data1[valid_idx]
    data2 = data2[valid_idx]
    
    if method == 'pearson':
        corr, p_value = stats.pearsonr(data1, data2)
    elif method == 'spearman':
        corr, p_value = stats.spearmanr(data1, data2)
    elif method == 'pointbiserial':
      
        if len(data1.unique()) == 2:
            corr, p_value = stats.pointbiserialr(data1, data2)
        elif len(data2.unique()) == 2:
            corr, p_value = stats.pointbiserialr(data2, data1)
        else:
            return jsonify({'error': 'Point-biserial requires one binary variable'})
    elif method == 'cramersv':
        
        contingency_table = pd.crosstab(data1, data2)
        chi2 = stats.chi2_contingency(contingency_table)[0]
        n = contingency_table.sum().sum()
        phi2 = chi2 / n
        r, k = contingency_table.shape
        corr = np.sqrt(phi2 / min((k-1), (r-1)))
        p_value = stats.chi2_contingency(contingency_table)[1]
    else:
        return jsonify({'error': 'Invalid correlation method'})
    
    return jsonify({
        'correlation': corr,
        'p_value': p_value,
        'method': method
    })

@app.route('/dimensionality_reduction', methods=['POST'])
def dimensionality_reduction():
    global global_df
    if global_df is None:
        return jsonify({'error': 'No data uploaded'})
    
    method = request.json.get('method')
    n_components = int(request.json.get('n_components', 2))
    columns = request.json.get('columns')
    
    if not all(col in global_df.columns for col in columns):
        return jsonify({'error': 'One or more columns not found'})
    
    numeric_df = global_df[columns].select_dtypes(include=['number']).dropna()
    
    if method == 'pca':
        reducer = PCA(n_components=n_components)
        result = reducer.fit_transform(numeric_df)
        explained_var = reducer.explained_variance_ratio_.tolist()
    elif method == 'tsne':
        perplexity = request.json.get('perplexity', 30)
        reducer = TSNE(n_components=n_components, perplexity=perplexity)
        result = reducer.fit_transform(numeric_df)
        explained_var = None
    elif method == 'umap':
        try:
            from umap import UMAP
            reducer = UMAP(n_components=n_components)
            result = reducer.fit_transform(numeric_df)
            explained_var = None
        except ImportError:
            return jsonify({'error': 'UMAP not installed. Install with: pip install umap-learn'})
    
    return jsonify({
        'result': result.tolist(),
        'explained_variance': explained_var,
        'columns': [f'Component {i+1}' for i in range(n_components)]
    })

@app.route('/transform', methods=['POST'])
def transform_data():
    global global_df
    if global_df is None:
        return jsonify({'error': 'No data uploaded'})
    
    column = request.json.get('column')
    method = request.json.get('method')
    
    if column not in global_df.columns:
        return jsonify({'error': 'Column not found'})
    
    data = global_df[column].dropna()
    
    if method == 'log':
        transformed = np.log1p(data)
    elif method == 'sqrt':
        transformed = np.sqrt(data)
    elif method == 'standardize':
        transformed = (data - data.mean()) / data.std()
    elif method == 'normalize':
        transformed = (data - data.min()) / (data.max() - data.min())
    elif method == 'bin':
        bins = int(request.json.get('bins', 5))
        transformed = pd.cut(data, bins=bins, labels=False)
    elif method == 'onehot':
       
        transformed = pd.get_dummies(data).to_dict(orient='records')
        return jsonify({'transformed': transformed})
    
    return jsonify({
        'original': data.tolist(),
        'transformed': transformed.tolist()
    })

@app.route('/cluster', methods=['POST'])
def cluster_data():
    global global_df
    if global_df is None:
        return jsonify({'error': 'No data uploaded'})
    
    method = request.json.get('method')
    columns = request.json.get('columns')
    n_clusters = int(request.json.get('n_clusters', 3))
    
    if not all(col in global_df.columns for col in columns):
        return jsonify({'error': 'One or more columns not found'})
    
    numeric_df = global_df[columns].select_dtypes(include=['number']).dropna()
    
    if method == 'kmeans':
        model = KMeans(n_clusters=n_clusters)
        clusters = model.fit_predict(numeric_df)
    elif method == 'hierarchical':
        from scipy.cluster.hierarchy import linkage, fcluster
        Z = linkage(numeric_df, method='ward')
        clusters = fcluster(Z, t=n_clusters, criterion='maxclust') - 1  
    elif method == 'dbscan':
        eps = float(request.json.get('eps', 0.5))
        min_samples = int(request.json.get('min_samples', 5))
        model = DBSCAN(eps=eps, min_samples=min_samples)
        clusters = model.fit_predict(numeric_df)
    
    return jsonify({
        'clusters': clusters.tolist(),
        'n_clusters': len(set(clusters)) - (1 if -1 in clusters else 0)
    })

@app.route('/association_rules', methods=['POST'])
def find_association_rules():
    global global_df
    if global_df is None:
        return jsonify({'error': 'No data uploaded'})
    
    min_support = float(request.json.get('min_support', 0.1))
    min_threshold = float(request.json.get('min_threshold', 0.5))
    columns = request.json.get('columns')
    
    if not all(col in global_df.columns for col in columns):
        return jsonify({'error': 'One or more columns not found'})
    
   
    data = global_df[columns].apply(lambda x: x.astype('category'))
    data_encoded = pd.get_dummies(data)
    
    
    frequent_itemsets = apriori(data_encoded, min_support=min_support, use_colnames=True)
    
    if frequent_itemsets.empty:
        return jsonify({'error': 'No frequent itemsets found with given support'})
    
    
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_threshold)
    
    
    rules['antecedents'] = rules['antecedents'].apply(lambda x: list(x))
    rules['consequents'] = rules['consequents'].apply(lambda x: list(x))
    
    return jsonify({
        'rules': rules.to_dict(orient='records'),
        'num_rules': len(rules)
    })

if __name__ == '__main__':
    app.run(debug=True, port=5002)