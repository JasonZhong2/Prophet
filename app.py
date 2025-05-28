from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
import os
import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde
import matplotlib
import logging
from logging.handlers import RotatingFileHandler
import joblib
import json
from pathlib import Path
import seaborn as sns
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from model_training import evaluate_direction_metrics, evaluate_predictions_by_timeframe

matplotlib.use('Agg')  # 设置后端为Agg
import matplotlib.pyplot as plt
from functools import wraps
import io
import base64

app = Flask(__name__)
app.secret_key = os.urandom(24)  # 设置密钥用于session

# 全局变量初始化
trained_models = {
    'prophet': None,
    'arima': None,
    'rf': None
}
model_metrics_data = None
prophet_plots = None
prophet_best_params = None  # 存储Prophet模型的最优参数

# 中国节假日定义
spring_festival = pd.DataFrame({
    'holiday': 'spring_festival',
    'ds': pd.to_datetime([
        '2013-02-10', '2014-01-31', '2015-02-19', '2016-02-08', '2017-01-28',
        '2018-02-16', '2019-02-05', '2020-01-25', '2021-02-12', '2022-02-01',
        '2023-01-22', '2024-02-10'
    ]),
    'lower_window': -3,
    'upper_window': 7
})

mid_autumn = pd.DataFrame({
    'holiday': 'mid_autumn',
    'ds': pd.to_datetime([
        '2013-09-19', '2014-09-08', '2015-09-27', '2016-09-15', '2017-10-04',
        '2018-09-24', '2019-09-13', '2020-10-01', '2021-09-21', '2022-09-10',
        '2023-09-29', '2024-09-17'
    ]),
    'lower_window': -1,
    'upper_window': 2
})

cn_holidays = pd.concat([spring_festival, mid_autumn], ignore_index=True)

# 全局变量存储训练好的模型和预测结果
prophet_model = None
prophet_forecast = None
model_metrics_results = None  # 存储模型指标对比结果
arima_model = None
rf_model = None

# 在全局变量初始化部分添加
available_stocks = {
    "600519.SH": "贵州茅台",
    "601328.SH": "交通银行"
}

# 配置日志
def setup_logger():
    # 创建logs目录（如果不存在）
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # 配置日志格式
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            RotatingFileHandler('logs/app.log', maxBytes=1024*1024, backupCount=5),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

# 创建logger实例
logger = setup_logger()

def load_trained_models(stock_code="600519.SH"):
    """加载预训练的模型和评估指标"""
    global trained_models, model_metrics_data, prophet_best_params
    try:
        # 检查必要的目录是否存在
        required_dirs = ['model_metrics', 'trained_models']
        for dir_name in required_dirs:
            if not os.path.exists(dir_name):
                raise FileNotFoundError(f"必要的目录不存在: {dir_name}")

        # 为每只股票创建独立的模型字典
        trained_models = {
            'prophet': None,
            'arima': None,
            'rf': None
        }

        # 加载Prophet最优参数
        prophet_params_path = Path(f'model_metrics/prophet_best_params_{stock_code.replace(".", "_")}.json')
        if prophet_params_path.exists():
            try:
                with open(prophet_params_path, 'r') as f:
                    prophet_best_params = json.load(f)
                logger.info(f"成功加载 {stock_code} 的Prophet最优参数: {prophet_best_params}")
            except Exception as e:
                logger.error(f"加载Prophet最优参数时出错: {str(e)}")
                prophet_best_params = {'changepoint_prior_scale': 0.01, 'seasonality_prior_scale': 20.0}
        else:
            logger.warning(f"Prophet最优参数文件不存在: {prophet_params_path}")
            prophet_best_params = {'changepoint_prior_scale': 0.01, 'seasonality_prior_scale': 20.0}
            
        # 加载评估指标
        metrics_path = Path(f'model_metrics/metrics_{stock_code.replace(".", "_")}.json')
        if metrics_path.exists():
            try:
                with open(metrics_path, 'r') as f:
                    model_metrics_data = json.load(f)
                logger.info(f"成功加载 {stock_code} 的模型评估指标: {model_metrics_data}")
            except Exception as e:
                logger.error(f"加载评估指标时出错: {str(e)}")
                model_metrics_data = None
        else:
            logger.warning(f"评估指标文件不存在: {metrics_path}")
            model_metrics_data = None

        # 加载模型
        stock_model_dir = f'trained_models/{stock_code.replace(".", "_")}'
        
        # 加载Prophet模型
        final_model_path = Path(f'{stock_model_dir}/prophet_model_final.joblib')
        if final_model_path.exists():
            trained_models['prophet'] = joblib.load(final_model_path)
            logger.info(f"加载 {stock_code} 的最终Prophet模型")
        else:
            raise FileNotFoundError(f"未找到 {stock_code} 的最终Prophet模型")

        # 加载ARIMA模型
        arima_model_path = Path(f'{stock_model_dir}/arima_model_final.joblib')
        if arima_model_path.exists():
            trained_models['arima'] = joblib.load(arima_model_path)
            logger.info(f"加载 {stock_code} 的ARIMA模型")
        else:
            raise FileNotFoundError(f"未找到 {stock_code} 的ARIMA模型")

        # 加载随机森林模型
        rf_model_path = Path(f'{stock_model_dir}/rf_model_final.joblib')
        if rf_model_path.exists():
            trained_models['rf'] = joblib.load(rf_model_path)
            logger.info(f"加载 {stock_code} 的随机森林模型")
        else:
            raise FileNotFoundError(f"未找到 {stock_code} 的随机森林模型")

        logger.info(f"成功加载 {stock_code} 的所有预训练模型")
        
    except Exception as e:
        logger.error(f"加载预训练模型时出错: {str(e)}")
        raise

def load_and_process_data(stock_code="600519.SH"):
    """直接读取已处理数据并划分"""
    input_file = f"processed_data_{stock_code.replace('.', '_')}.csv"
    data = pd.read_csv(input_file, index_col='trade_date', parse_dates=True)
    
    # 确保索引是datetime类型
    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index)
    
    # 数据集划分
    train_data = data[(data.index >= '2013-01-01') & (data.index <= '2022-12-31')]
    val_data = data[(data.index >= '2023-01-01') & (data.index <= '2023-12-31')]
    test_data = data[(data.index >= '2024-01-01') & (data.index <= '2024-12-31')]
    full_data = pd.concat([train_data, val_data, test_data])
    
    return {
        'train': train_data,
        'val': val_data,
        'test': test_data,
        'full': full_data
    }

# 登录验证装饰器
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'logged_in' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)

    return decorated_function


# 主页路由
@app.route('/')
def index():
    if 'logged_in' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))


# 登录路由
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if username == 'admin' and password == 'admin123':
            session['logged_in'] = True
            return redirect(url_for('dashboard'))
        else:
            flash('用户名或密码错误！')

    return render_template('login.html')


# 仪表板路由
@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html')


# 基础分析API端点
@app.route('/api/basic-analysis', methods=['GET'])
@login_required
def basic_analysis():
    try:
        stock_code = request.args.get('stock_code', '600519.SH')
        if stock_code not in available_stocks:
            raise ValueError(f"不支持的股票代码: {stock_code}")
            
        logger.info(f"开始 {available_stocks[stock_code]} 的基础分析...")
        data_dict = load_and_process_data(stock_code)
        data = data_dict['full']

        # 设置matplotlib中文字体和样式
        plt.style.use('seaborn-v0_8')
        plt.rcParams["font.sans-serif"] = ["SimHei"]
        plt.rcParams["axes.unicode_minus"] = False

        # 生成趋势图
        logger.info("生成趋势图...")
        fig1 = plt.figure(figsize=(8, 4))
        plt.plot(data.index, data['close'], 'k-', linewidth=1, alpha=0.7)
        plt.scatter(data.index, data['close'], c='blue', s=5, alpha=0.5)
        plt.title(f'{available_stocks[stock_code]}股票收盘价趋势（2013年-2024年）', fontsize=12, pad=10)
        plt.xlabel('日期', fontsize=10)
        plt.ylabel('收盘价 (RMB)', fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(rotation=45, fontsize=8)
        plt.yticks(fontsize=8)
        plt.tight_layout()
        trend_url = get_plot_url(fig1)

        # 生成分布图
        logger.info("生成分布图...")
        fig2 = plt.figure(figsize=(6, 4))
        kde = gaussian_kde(data['close'])
        x_range = np.linspace(data['close'].min(), data['close'].max(), 100)
        density = kde(x_range)
        plt.hist(data['close'], bins=30, density=True, alpha=0.6, color='skyblue', edgecolor='black')
        plt.plot(x_range, density, 'r-', linewidth=1.5, label='密度曲线')
        plt.title(f'{available_stocks[stock_code]}收盘价分布（2013年-2024年）', fontsize=12, pad=10)
        plt.xlabel('收盘价 (RMB)', fontsize=10)
        plt.ylabel('密度', fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=8)
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)
        plt.tight_layout()
        dist_url = get_plot_url(fig2)

        if not trend_url or not dist_url:
            raise Exception("图片生成失败")

        logger.info("基础分析完成")

        # 计算基本统计信息
        stats = {
            'mean': float(data['close'].mean()),
            'std': float(data['close'].std()),
            'min': float(data['close'].min()),
            'max': float(data['close'].max()),
            'latest': float(data['close'].iloc[-1]),
            'start_date': data.index.min().strftime('%Y-%m-%d'),
            'end_date': data.index.max().strftime('%Y-%m-%d'),
            'total_days': len(data)
        }

        return jsonify({
            'stock_name': available_stocks[stock_code],
            'trend_plot': trend_url,
            'distribution_plot': dist_url,
            'stats': stats
        })
    except Exception as e:
        logger.error(f"基础分析错误: {str(e)}")
        return jsonify({'error': str(e)}), 500


# 在app初始化时加载模型
load_trained_models()

# 修改Prophet预测API端点
@app.route('/api/prophet-prediction', methods=['GET'])
@login_required
def prophet_prediction():
    try:
        stock_code = request.args.get('stock_code', '600519.SH')
        if stock_code not in available_stocks:
            raise ValueError(f"不支持的股票代码: {stock_code}")
            
        logger.info(f"开始 {available_stocks[stock_code]} 的Prophet预测...")
        global prophet_best_params
        
        # 加载对应股票的模型
        load_trained_models(stock_code)
        
        data_dict = load_and_process_data(stock_code)
        data = data_dict['full']
        train_data = data_dict['train']
        val_data = data_dict['val']
        test_data = data_dict['test']

        if trained_models['prophet'] is None:
            raise ValueError(f"未找到 {available_stocks[stock_code]} 的预训练Prophet模型")
        model = trained_models['prophet']

        end_date = pd.to_datetime('2025-12-31')
        periods = (end_date - data.index[-1]).days
        if periods < 1:
            periods = 1
        future = model.make_future_dataframe(periods=periods)
        prophet_forecast = model.predict(future)

        # 只保留测试集误差分布直方图+密度曲线
        test_idx = test_data.index
        test_pred = prophet_forecast[prophet_forecast['ds'].isin(test_idx)]
        # 修改误差计算方法，确保与ARIMA和RF模型一致
        error_test = test_data['close'].values - test_pred['yhat'].values if not test_pred.empty else np.array([])
        fig_test_err = plt.figure(figsize=(8,4))
        if len(error_test) > 0:
            plt.hist(error_test, bins=30, color='skyblue', edgecolor='black', alpha=0.6, density=True)
            sns.kdeplot(error_test, color='red', linewidth=2, label='密度曲线')
        plt.title('Prophet预测误差分布')
        plt.xlabel('误差')
        plt.ylabel('密度')
        plt.grid(True)
        plt.legend()
        test_error_hist_url = get_plot_url(fig_test_err)

        # 计算方向性指标
        y_true = test_data['close'].values
        y_pred = test_pred['yhat'].values if not test_pred.empty else np.array([])
        prophet_direction = evaluate_direction_metrics(y_true, y_pred)

        # 检查预测结果
        logger.info(f"Prophet预测结果范围: {prophet_forecast['ds'].min()} 到 {prophet_forecast['ds'].max()}")
        logger.info(f"预测结果行数: {len(prophet_forecast)}")

        # 在生成预测图之前添加
        logger.info(f"历史数据行数: {len(data)}")
        logger.info(f"预测数据行数: {len(prophet_forecast)}")

        # 1. 预测图（Prophet默认风格，颜色与趋势分解图一致）
        fig1, ax = plt.subplots(figsize=(14, 7))
        ax.scatter(data.index, data['close'], color='black', s=10, label='历史数据')
        ax.plot(prophet_forecast['ds'], prophet_forecast['yhat'], color='#0072B2', label='预测值', linewidth=2)
        ax.set_title(f'{available_stocks[stock_code]}股票价格预测')
        ax.set_xlabel('日期')
        ax.set_ylabel('收盘价 (RMB)')
        ax.legend()
        ax.grid(True)
        forecast_url = get_plot_url(fig1)

        # 2. 趋势分解图（到2025年）
        fig2 = model.plot_components(prophet_forecast)
        for ax2 in fig2.get_axes():
            ax2.legend()
            ax2.grid(True)
        components_url = get_plot_url(fig2)

        # 3. 实际值与预测值对比图（全预测线，2013~2025）
        fig3 = plt.figure(figsize=(14, 7))
        plt.plot(data.index, data['close'], label='实际值', color='blue', alpha=0.7)
        plt.plot(prophet_forecast['ds'], prophet_forecast['yhat'], label='预测值', color='orange', linestyle='--', alpha=0.9)
        plt.title('实际值与预测值对比')
        plt.xlabel('日期')
        plt.ylabel('收盘价 (RMB)')
        plt.legend()
        plt.grid(True)
        compare_url = get_plot_url(fig3)

        return jsonify({
            'stock_name': available_stocks[stock_code],
            'forecast_plot': forecast_url,
            'components_plot': components_url,
            'compare_plot': compare_url,
            'parameters': {
                'changepoint_prior_scale': prophet_best_params['changepoint_prior_scale'],
                'seasonality_prior_scale': prophet_best_params['seasonality_prior_scale']
            },
            'test_error_hist': test_error_hist_url,
            'direction_metrics': prophet_direction
        })

    except Exception as e:
        logger.error(f"Prophet预测错误: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500


# 修改模型评估API端点
@app.route('/api/model-metrics', methods=['GET'])
@login_required
def model_metrics():
    try:
        stock_code = request.args.get('stock_code', '600519.SH')
        if stock_code not in available_stocks:
            raise ValueError(f"不支持的股票代码: {stock_code}")
            
        logger.info(f"请求 {available_stocks[stock_code]} 的模型评估指标...")
        
        # 加载对应股票的模型和指标
        load_trained_models(stock_code)
        
        logger.info("正在生成模型评估指标对比图...")
        logger.info(f"使用的模型指标: {model_metrics_data}")

        # 生成评估指标对比图
        metrics = ['rmse', 'mae', 'mape']
        models = ['prophet', 'arima', 'rf']
        model_names = ['Prophet', 'ARIMA', 'Random Forest']

        # 确保所有需要的模型和指标都存在
        for model in models:
            if model not in model_metrics_data:
                raise ValueError(f"模型指标中缺少模型: {model}")
            if 'test' not in model_metrics_data[model]:
                raise ValueError(f"模型 {model} 缺少测试集指标(test)")
            for metric in metrics:
                if metric not in model_metrics_data[model]['test']:
                    raise ValueError(f"模型 {model} 的测试集指标中缺少: {metric}")

        # 生成RMSE指标对比图
        fig_rmse = plt.figure(figsize=(10, 6))
        rmse_values = [model_metrics_data[model]['test']['rmse'] for model in models]
        bars = plt.bar(model_names, rmse_values)
        plt.title('RMSE指标对比（测试集）')
        plt.xlabel('模型')
        plt.ylabel('RMSE值')
        bars[0].set_color('orange')
        for i, v in enumerate(rmse_values):
            plt.text(i, v, f'{v:.2f}', ha='center', va='bottom')
        plt.tight_layout()
        rmse_metrics_url = get_plot_url(fig_rmse)

        # 生成MAE指标对比图
        fig_mae = plt.figure(figsize=(10, 6))
        mae_values = [model_metrics_data[model]['test']['mae'] for model in models]
        bars = plt.bar(model_names, mae_values)
        plt.title('MAE指标对比（测试集）')
        plt.xlabel('模型')
        plt.ylabel('MAE值')
        bars[0].set_color('orange')
        for i, v in enumerate(mae_values):
            plt.text(i, v, f'{v:.2f}', ha='center', va='bottom')
        plt.tight_layout()
        mae_metrics_url = get_plot_url(fig_mae)

        # 生成MAPE指标对比图
        fig_mape = plt.figure(figsize=(10, 6))
        mape_values = [model_metrics_data[model]['test']['mape'] for model in models]
        bars = plt.bar(model_names, mape_values)
        plt.title('MAPE指标对比（测试集）')
        plt.xlabel('模型')
        plt.ylabel('MAPE值')
        bars[0].set_color('orange')
        for i, v in enumerate(mape_values):
            plt.text(i, v, f'{v:.2%}', ha='center', va='bottom')
        plt.tight_layout()
        mape_metrics_url = get_plot_url(fig_mape)

        # 找出最佳模型
        best_model = min(model_metrics_data.items(), key=lambda x: x[1]['test']['rmse'])[0]
        best_model_name = model_names[models.index(best_model)]

        logger.info(f"返回评估指标，最佳模型: {best_model_name}")
        # 提取方向性指标
        direction = {model: model_metrics_data[model].get('direction_metrics', {}) for model in models}
        return jsonify({
            'stock_name': available_stocks[stock_code],
            'rmse_metrics_plot': rmse_metrics_url,
            'mae_metrics_plot': mae_metrics_url,
            'mape_metrics_plot': mape_metrics_url,
            'model_comparison': {
                'best_model': best_model_name,
                'metrics': {
                    'rmse': model_metrics_data[best_model]['test']['rmse'],
                    'mae': model_metrics_data[best_model]['test']['mae'],
                    'mape': model_metrics_data[best_model]['test']['mape']
                }
            },
            'direction_metrics': direction  # 添加方向性指标
        })

    except Exception as e:
        logger.error(f"模型评估错误: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500


def get_plot_url(fig):
    try:
        # 保存图片到内存
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1, dpi=150)
        plt.close(fig)  # 关闭图片以释放内存

        # 转换为base64
        buf.seek(0)
        image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        buf.close()

        if not image_base64:
            raise ValueError("图片转换为base64失败")

        result = f'data:image/png;base64,{image_base64}'
        logger.debug(f"生成的图片URL前20个字符: {result[:20]}...")
        return result

    except Exception as e:
        logger.error(f"生成图片URL时发生错误: {str(e)}")
        return None


# 登出路由
@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    return redirect(url_for('login'))


# 修改ARIMA分析API端点
@app.route('/api/arima-analysis', methods=['GET'])
@login_required
def arima_analysis():
    try:
        # 根据请求参数获取股票代码
        stock_code = request.args.get('stock_code', '600519.SH')
        if stock_code not in available_stocks:
            raise ValueError(f"不支持的股票代码: {stock_code}")
        logger.info(f"开始 {available_stocks[stock_code]} 的ARIMA分析...")
        # 加载对应股票的模型和数据
        load_trained_models(stock_code)
        arima_data = load_and_process_data(stock_code)
        arima_model = trained_models['arima']['model']
        last_price = trained_models['arima']['last_price']
        train_data = arima_data['train']
        val_data = arima_data['val']
        test_data = arima_data['test']

        # 训练集预测（仅用于计算，不显示）
        train_forecast_returns = arima_model.forecast(steps=len(train_data))
        train_forecast_prices = last_price * np.exp(np.cumsum(train_forecast_returns))
        train_forecast_prices.index = train_data.index

        # 验证集预测
        val_forecast_returns = arima_model.forecast(steps=len(val_data))
        val_forecast_prices = last_price * np.exp(np.cumsum(val_forecast_returns))
        val_forecast_prices.index = val_data.index

        # 测试集预测
        test_forecast_returns = arima_model.forecast(steps=len(test_data))
        test_forecast_prices = last_price * np.exp(np.cumsum(test_forecast_returns))
        test_forecast_prices.index = test_data.index

        # 误差分布图（只针对测试集）
        error_test = test_data['close'] - test_forecast_prices
        fig_test_err = plt.figure(figsize=(8,4))
        plt.hist(error_test, bins=30, color='skyblue', edgecolor='black', alpha=0.6, density=True)
        try:
            sns.kdeplot(error_test, color='red', linewidth=2, label='密度曲线')
        except Exception as e:
            logger.warning(f'seaborn.kdeplot 画密度曲线失败: {e}')
        plt.title('ARIMA预测误差分布')
        plt.xlabel('误差')
        plt.ylabel('密度')
        plt.grid(True)
        plt.legend()
        test_error_hist_url = get_plot_url(fig_test_err)

        # 计算方向性指标
        arima_direction = evaluate_direction_metrics(test_data['close'].values, test_forecast_prices.values)
        # 合并训练集和验证集
        trainval_data = pd.concat([train_data, val_data])
        # ARIMA分析API端点
        fig1 = plt.figure(figsize=(16, 7))
        plt.plot(trainval_data.index, trainval_data['close'], color='blue', label='训练+验证集实际值')
        plt.plot(test_data.index, test_data['close'], color='green', label='测试集实际值')
        plt.plot(test_data.index, test_forecast_prices, color='red', linestyle='--', label='测试集预测', linewidth=2, alpha=0.8)
        plt.title('ARIMA模型测试集预测结果')
        plt.xlabel('日期')
        plt.ylabel('收盘价 (RMB)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        prediction_url = get_plot_url(fig1)

        return jsonify({
            'prediction_plot': prediction_url,
            'test_error_hist': test_error_hist_url,
            'direction_metrics': arima_direction
        })
    except Exception as e:
        logger.error(f"ARIMA分析错误: {str(e)}")
        return jsonify({'error': str(e)}), 500


# 修改随机森林分析API端点
@app.route('/api/random-forest-analysis', methods=['GET'])
@login_required
def random_forest_analysis():
    try:
        # 根据请求参数获取股票代码
        stock_code = request.args.get('stock_code', '600519.SH')
        if stock_code not in available_stocks:
            raise ValueError(f"不支持的股票代码: {stock_code}")
        logger.info(f"开始 {available_stocks[stock_code]} 的随机森林分析...")
        # 加载对应股票的模型和数据
        load_trained_models(stock_code)
        data_dict = load_and_process_data(stock_code)
        train_data = data_dict['train']
        val_data = data_dict['val']
        test_data = data_dict['test']
        # 确保模型已加载（load_trained_models 已处理）

        # 准备特征和预测（训练集预测仅用于计算，不显示）
        train_features = prepare_rf_features(train_data)
        val_features = prepare_rf_features(val_data)
        test_features = prepare_rf_features(test_data)

        # 添加日志验证数据
        logger.info(f"验证集数据范围: {val_data.index.min()} 到 {val_data.index.max()}")
        logger.info(f"验证集数据点数: {len(val_data)}")

        rf_model = trained_models['rf']
        train_pred = rf_model.predict(train_features)
        val_pred = rf_model.predict(val_features)
        test_pred = rf_model.predict(test_features)

        # 添加日志验证预测值
        logger.info(f"验证集预测值范围: {val_pred.min():.2f} 到 {val_pred.max():.2f}")
        logger.info(f"验证集预测值数量: {len(val_pred)}")
        logger.info(f"验证集实际值范围: {val_data['close'].min():.2f} 到 {val_data['close'].max():.2f}")

        # 验证集预测误差
        val_error = val_data['close'] - val_pred
        logger.info(f"验证集预测误差统计:")
        logger.info(f"RMSE: {np.sqrt(np.mean(val_error**2)):.2f}")
        logger.info(f"MAE: {np.mean(np.abs(val_error)):.2f}")

        # 误差分布图（只针对测试集）
        error_test = test_data['close'] - test_pred
        fig_test_err = plt.figure(figsize=(8,4))
        plt.hist(error_test, bins=30, color='skyblue', edgecolor='black', alpha=0.6, density=True)
        sns.kdeplot(error_test, color='red', linewidth=2, label='密度曲线')
        plt.title('随机森林预测误差分布')
        plt.xlabel('误差')
        plt.ylabel('密度')
        plt.grid(True)
        plt.legend()
        test_error_hist_url = get_plot_url(fig_test_err)
        
        # 计算方向性指标
        rf_direction = evaluate_direction_metrics(test_data['close'].values, test_pred)
        
        # 合并训练集和验证集
        trainval_data = pd.concat([train_data, val_data])
        # 随机森林分析API端点
        fig1 = plt.figure(figsize=(16, 7))
        plt.plot(trainval_data.index, trainval_data['close'], color='blue', label='训练+验证集实际值', alpha=0.7)
        plt.plot(test_data.index, test_data['close'], color='green', label='测试集实际值', alpha=0.7)
        plt.plot(test_data.index, test_pred, color='red', linestyle='--', label='测试集预测', linewidth=2, alpha=0.8)
        plt.title('随机森林模型测试集预测结果')
        plt.xlabel('日期')
        plt.ylabel('收盘价 (RMB)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        prediction_url = get_plot_url(fig1)
        
        # 特征重要性图
        feature_names = test_features.columns.tolist()
        importances = trained_models['rf'].feature_importances_
        fig_fi = plt.figure(figsize=(8, 5))
        plt.bar(feature_names, importances, color='skyblue')
        plt.title('随机森林特征重要性')
        plt.xlabel('特征')
        plt.ylabel('重要性')
        plt.tight_layout()
        feature_importance_url = get_plot_url(fig_fi)
        
        return jsonify({
            'prediction_plot': prediction_url,
            'test_error_hist': test_error_hist_url,
            'feature_importance_plot': feature_importance_url,
            'direction_metrics': rf_direction
        })
    except Exception as e:
        logger.error(f"随机森林分析错误: {str(e)}")
        return jsonify({'error': str(e)}), 500


def prepare_rf_features(data):
    """准备随机森林模型的特征"""
    features = pd.DataFrame({
        'year': data.index.year,
        'month': data.index.month,
        'day': data.index.day,
        'dayofweek': data.index.dayofweek
    })
    return features


# 添加新的API端点用于获取所有股票的对比结果
@app.route('/api/stocks-comparison', methods=['GET'])
@login_required
def stocks_comparison():
    try:
        logger.info("获取所有股票的模型对比结果...")
        
        # 读取所有股票的对比结果
        comparison_file = 'model_metrics/all_stocks_comparison.json'
        if not os.path.exists(comparison_file):
            raise FileNotFoundError("未找到股票对比结果文件")
            
        with open(comparison_file, 'r') as f:
            all_stocks_metrics = json.load(f)
            
        # 生成对比图表
        metrics = ['rmse', 'mae', 'mape']
        models = ['prophet', 'arima', 'rf']
        model_names = ['Prophet', 'ARIMA', 'Random Forest']
        
        comparison_plots = {}
        for metric in metrics:
            fig = plt.figure(figsize=(12, 6))
            x = np.arange(len(available_stocks))
            width = 0.25
            
            for i, (model, model_name) in enumerate(zip(models, model_names)):
                values = [all_stocks_metrics[stock_code][model]['test'][metric] 
                         for stock_code in available_stocks.keys()]
                bars = plt.bar(x + i*width, values, width, label=model_name)
                # 添加数值标签
                for j, v in enumerate(values):
                    label = f'{v:.2f}' if metric != 'mape' else f'{v:.2%}'
                    plt.text(x[j] + i*width, v, label, ha='center', va='bottom')
            
            plt.title(f'各股票{metric.upper()}指标对比')
            plt.xlabel('股票')
            plt.ylabel(metric.upper())
            plt.xticks(x + width, [available_stocks[code] for code in available_stocks.keys()])
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            
            comparison_plots[f'{metric}_comparison'] = get_plot_url(fig)
            
        return jsonify({
            'comparison_plots': comparison_plots,
            'stocks_metrics': all_stocks_metrics
        })
        
    except Exception as e:
        logger.error(f"获取股票对比结果时出错: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/timeframe-predictions', methods=['GET'])
@login_required
def timeframe_predictions():
    try:
        # 获取股票代码参数
        stock_code = request.args.get('stock_code', '600519.SH')
        if stock_code not in available_stocks:
            raise ValueError(f"不支持的股票代码: {stock_code}")
        # 加载对应股票的模型和数据
        load_trained_models(stock_code)
        data_dict = load_and_process_data(stock_code)
        full_data = data_dict['full']
        # 获取最终Prophet模型
        model = trained_models['prophet']
        if model is None:
            raise ValueError(f"未找到 {available_stocks[stock_code]} 的Prophet模型")
        # 动态生成多时间尺度预测结果
        results = evaluate_predictions_by_timeframe(model, full_data, stock_code)
        return jsonify(results)
    except Exception as e:
        logger.error(f"多时间尺度预测接口错误: {str(e)}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    logger.info("正在启动股票分析系统...")
    try:
        logger.info("加载预训练模型和指标...")
        # 默认加载贵州茅台的模型
        load_trained_models("600519.SH")
        logger.info(f"模型指标加载状态: {model_metrics_data is not None}")
        logger.info(f"已加载的模型: {[k for k, v in trained_models.items() if v is not None]}")
        app.run(debug=True)
    except Exception as e:
        logger.error(f"应用启动错误: {str(e)}", exc_info=True)
