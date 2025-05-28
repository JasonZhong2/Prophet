import pandas as pd
import numpy as np
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import joblib
import os
import json
from datetime import timedelta
import logging
from tqdm import tqdm
import time
import sys
import glob
from itertools import product
import matplotlib.pyplot as plt
import io, base64
import importlib.util
import matplotlib.dates as mdates
import datetime
from dateutil.relativedelta import relativedelta

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/model_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

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



def load_and_split_data(stock_code):
    """加载已处理数据并划分训练、验证、测试集"""
    input_file = f"processed_data_{stock_code.replace('.', '_')}.csv"
    data = pd.read_csv(input_file, index_col='trade_date', parse_dates=True)
    train_data = data[(data.index >= '2013-01-01') & (data.index <= '2022-12-31')]
    val_data = data[(data.index >= '2023-01-01') & (data.index <= '2023-12-31')]
    test_data = data[(data.index >= '2024-01-01') & (data.index <= '2024-12-31')]
    return train_data, val_data, test_data


def rolling_window_validation(data, window_size=1200, step_size=240):
    """实现滚动窗口交叉验证"""
    start_date = data.index.min()
    end_date = data.index.max()
    windows = []

    current_start = start_date
    while current_start + timedelta(days=window_size) <= end_date:
        current_end = current_start + timedelta(days=window_size)
        windows.append((current_start, current_end))
        current_start += timedelta(days=step_size)

    return windows


def optimize_prophet_parameters(train_data, val_data):
    """只对changepoint_prior_scale和seasonality_prior_scale做网格搜索，启用全部季节性分量"""
    changepoint_prior_scales = [0.01, 0.05, 0.1, 0.2]
    seasonality_prior_scales = [1.0, 5.0, 10.0, 20.0]

    best_params = None
    best_rmse = float('inf')
    results = []

    # 准备训练数据
    train_df = train_data.reset_index()[['trade_date', 'close']].copy()
    train_df.columns = ['ds', 'y']

    # 准备验证数据
    val_df = val_data.reset_index()[['trade_date', 'close']].copy()
    val_df.columns = ['ds', 'y']

    for cp_scale in changepoint_prior_scales:
        for season_scale in seasonality_prior_scales:
            try:
                model = Prophet(
                    changepoint_prior_scale=cp_scale,
                    seasonality_prior_scale=season_scale,
                    yearly_seasonality=False,  # 禁用默认年度季节性
                    weekly_seasonality=False,  # 禁用默认周季节性
                    daily_seasonality=False,
                    holidays=cn_holidays,
                    mcmc_samples=0
                )
                model.add_seasonality(name='custom_yearly', period=240, fourier_order=10)      # 年度季节性
                model.add_seasonality(name='custom_monthly', period=20, fourier_order=3)       # 月度季节性
                model.add_seasonality(name='custom_weekly', period=5, fourier_order=3)         # 周度季节性
                model.add_country_holidays(country_name='CN')
                model.fit(train_df)

                forecast = model.predict(val_df[['ds']])
                rmse = np.sqrt(mean_squared_error(val_df['y'], forecast['yhat']))
                mae = mean_absolute_error(val_df['y'], forecast['yhat'])
                mape = mean_absolute_percentage_error(val_df['y'], forecast['yhat'])

                results.append({
                    'changepoint_prior_scale': cp_scale,
                    'seasonality_prior_scale': season_scale,
                    'rmse': rmse,
                    'mae': mae,
                    'mape': mape
                })

                if rmse < best_rmse:
                    best_rmse = rmse
                    best_params = {
                        'changepoint_prior_scale': cp_scale,
                        'seasonality_prior_scale': season_scale,
                        'yearly_seasonality': False,
                        'weekly_seasonality': False,
                        'daily_seasonality': False,
                        'quarterly_seasonality': True,
                        'monthly_seasonality': True
                    }
            except Exception as e:
                logger.error(f"参数组合 (cp={cp_scale}, season={season_scale}) 训练失败: {str(e)}")
                continue

    if best_params is None:
        raise ValueError("没有找到有效的参数组合")

    # 保存参数优化结果
    results_df = pd.DataFrame(results)
    results_df.to_csv('model_metrics/prophet_parameter_optimization.csv', index=False)
    with open('model_metrics/prophet_best_params.json', 'w') as f:
        json.dump(best_params, f, indent=4)
    logger.info(f"保存Prophet最优参数: {best_params}")
    return best_params, results_df


def train_prophet_model(train_data, params):
    """使用优化后的参数训练Prophet模型"""
    try:
        # 准备训练数据
        train_df = train_data.reset_index()[['trade_date', 'close']].copy()
        train_df.columns = ['ds', 'y']

        model = Prophet(
            changepoint_prior_scale=params['changepoint_prior_scale'],
            seasonality_prior_scale=params['seasonality_prior_scale'],
            yearly_seasonality=False,  # 禁用默认年度季节性
            weekly_seasonality=False,  # 禁用默认周季节性
            daily_seasonality=params.get('daily_seasonality', False),
            holidays=cn_holidays,
            mcmc_samples=0
        )
        model.add_seasonality(name='custom_yearly', period=240, fourier_order=10)
        model.add_seasonality(name='custom_monthly', period=20, fourier_order=3)
        model.add_seasonality(name='custom_weekly', period=5, fourier_order=3)
        model.add_country_holidays(country_name='CN')
        model.fit(train_df)
        return model
    except Exception as e:
        logger.error(f"Prophet模型训练错误: {str(e)}")
        raise


def train_arima_model(train_data):
    """训练ARIMA模型"""
    try:
        # 确保数据不为空
        if train_data.empty:
            raise ValueError("训练数据为空")

        # 确保log_return列存在且不为空
        if 'log_return' not in train_data.columns or train_data['log_return'].isna().all():
            train_data['log_return'] = np.log(train_data['close']).diff().dropna()

        # 使用预处理后的对数收益率数据
        model = ARIMA(train_data['log_return'].dropna(),
                      order=(1, 1, 1),  # (p,d,q)
                      seasonal_order=(1, 1, 1, 12))  # (P,D,Q,s)
        fitted_model = model.fit()

        # 确保返回最后一个价格
        last_price = train_data['close'].iloc[-1]
        if pd.isna(last_price):
            raise ValueError("最后一个价格为空")

        return fitted_model, last_price
    except Exception as e:
        logger.error(f"ARIMA模型训练错误: {str(e)}")
        raise


def train_rf_model(train_features, train_target):
    """训练随机森林模型，使用固定参数"""
    try:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(train_features, train_target)
        return model
    except Exception as e:
        logger.error(f"随机森林模型训练错误: {str(e)}")
        raise


def evaluate_model(y_true, y_pred):
    """计算模型评估指标"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    return {'rmse': rmse, 'mae': mae, 'mape': mape}


def prepare_rf_features(data):
    """准备随机森林模型的特征"""
    features = pd.DataFrame({
        'year': data.index.year,
        'month': data.index.month,
        'day': data.index.day,
        'dayofweek': data.index.dayofweek
    })
    return features


def check_training_completion():
    """检查模型训练是否完成"""
    try:
        # 检查必要的目录和文件是否存在
        if not os.path.exists('trained_models'):
            return False, "模型目录不存在"

        if not os.path.exists('model_metrics/metrics.json'):
            return False, "评估指标文件不存在"

        # 检查模型文件
        model_files = {
            'prophet': list(glob.glob('trained_models/prophet_model_*.joblib')),
            'arima': list(glob.glob('trained_models/arima_model_*.joblib')),
            'rf': list(glob.glob('trained_models/rf_model_*.joblib'))
        }

        # 检查每个模型是否都有训练好的文件
        for model_name, files in model_files.items():
            if not files:
                return False, f"{model_name}模型文件不存在"

        # 检查评估指标文件内容
        with open('model_metrics/metrics.json', 'r') as f:
            metrics = json.load(f)
            if not all(model in metrics for model in ['prophet', 'arima', 'rf']):
                return False, "评估指标文件不完整"

        return True, "所有模型训练完成"

    except Exception as e:
        return False, f"检查训练完成状态时出错: {str(e)}"


def print_completion_message(metrics, training_time):
    """打印训练完成消息"""
    print("\n" + "=" * 80)
    print(" " * 20 + "模型训练和评估完成！" + " " * 20)
    print("=" * 80)

    print("\n训练结果摘要:")
    print("-" * 80)
    print(f"总训练时间: {training_time:.2f} 分钟")
    print(f"模型保存位置: {os.path.abspath('trained_models')}")
    print(f"评估指标保存位置: {os.path.abspath('model_metrics/metrics.json')}")
    print("-" * 80)

    # 读取并显示Prophet参数优化结果
    try:
        prophet_params = pd.read_csv('model_metrics/prophet_parameter_optimization.csv')
        best_params = prophet_params.loc[prophet_params['rmse'].idxmin()]
        print("\nProphet模型参数优化结果（基于验证集2023年数据）:")
        print("-" * 80)
        print(f"changepoint_prior_scale: {best_params['changepoint_prior_scale']}")
        print(f"seasonality_prior_scale: {best_params['seasonality_prior_scale']}")
        print(f"最优RMSE: {best_params['rmse']:.2f}")
        print(f"最优MAE: {best_params['mae']:.2f}")
        print(f"最优MAPE: {best_params['mape']:.2%}")
        print("-" * 80)
    except Exception as e:
        logger.error(f"读取Prophet参数优化结果时出错: {str(e)}")

    print("\n最终模型评估指标（验证集与测试集）:")
    print("-" * 80)
    for model_name, model_metrics in metrics.items():
        if model_name == 'prophet':
            print(f"\n{model_name.upper()}模型:")
            print(f"  验证集  RMSE: {model_metrics['val']['rmse']:.2f}  MAE: {model_metrics['val']['mae']:.2f}  MAPE: {model_metrics['val']['mape']:.2%}")
            print(f"  测试集  RMSE: {model_metrics['test']['rmse']:.2f}  MAE: {model_metrics['test']['mae']:.2f}  MAPE: {model_metrics['test']['mape']:.2%}")
        else:
            print(f"\n{model_name.upper()}模型:")
            print(f"  最终指标  RMSE: {model_metrics['test']['rmse']:.2f}  MAE: {model_metrics['test']['mae']:.2f}  MAPE: {model_metrics['test']['mape']:.2%}")

    print("-" * 80)

    print("\n指标说明:")
    print("1. 参数优化结果：在验证集（2023年数据）上使用不同参数组合测试得到的最优结果")
    print("2. Prophet模型验证集指标：滚动窗口交叉验证的平均指标；测试集指标：最终模型在2024年测试集上的真实泛化能力")
    print("3. ARIMA和RF模型最终指标：最终模型在2024年测试集上的真实泛化能力")
    print("-" * 80)

    print("\n方向性预测指标:")
    print("-" * 80)
    for model_name, model_metrics in metrics.items():
        if 'direction_metrics' in model_metrics:
            print(f"\n{model_name.upper()}模型方向性指标:")
            print(f"  总体方向准确率: {model_metrics['direction_metrics']['direction_accuracy']:.2%}")
            print(f"  上涨预测准确率: {model_metrics['direction_metrics']['up_accuracy']:.2%}")
            print(f"  下跌预测准确率: {model_metrics['direction_metrics']['down_accuracy']:.2%}")
            print(f"  最大连续正确预测: {model_metrics['direction_metrics']['max_trend_streak']} 天")
            print(f"  平均趋势持续期: {model_metrics['direction_metrics']['avg_trend_period']:.1f} 天")
    
    print("-" * 80)

    print("\n下一步操作:")
    print("1. 检查 trained_models 目录中的模型文件")
    print("2. 查看 model_metrics/metrics.json 中的详细评估指标")
    print("3. 运行 app.py 启动预测系统")
    print("=" * 80 + "\n")


def evaluate_direction_metrics(y_true, y_pred):
    """
    评估预测的方向性指标
    
    参数:
        y_true: 实际值
        y_pred: 预测值
    
    返回:
        dict: 包含方向性评估指标
    """
    try:
        # 计算实际价格变化方向
        true_direction = np.sign(np.diff(y_true))
        # 计算预测价格变化方向
        pred_direction = np.sign(np.diff(y_pred))
        
        # 方向准确率（预测方向与实际方向一致的百分比）
        direction_accuracy = np.mean(true_direction == pred_direction)
        
        # 上涨预测准确率
        up_mask = true_direction > 0
        up_accuracy = np.mean(true_direction[up_mask] == pred_direction[up_mask]) if np.any(up_mask) else 0
        
        # 下跌预测准确率
        down_mask = true_direction < 0
        down_accuracy = np.mean(true_direction[down_mask] == pred_direction[down_mask]) if np.any(down_mask) else 0
        
        # 趋势判断能力（连续正确预测的比率）
        correct_predictions = (true_direction == pred_direction)
        trend_streak = 0
        max_trend_streak = 0
        for is_correct in correct_predictions:
            if is_correct:
                trend_streak += 1
                max_trend_streak = max(max_trend_streak, trend_streak)
            else:
                trend_streak = 0
        
        # 计算平均趋势持续期
        trend_periods = []
        current_trend = 0
        for i in range(1, len(true_direction)):
            if true_direction[i] == true_direction[i-1]:
                current_trend += 1
            else:
                if current_trend > 0:
                    trend_periods.append(current_trend)
                current_trend = 0
        if current_trend > 0:
            trend_periods.append(current_trend)
        avg_trend_period = np.mean(trend_periods) if trend_periods else 0
        
        return {
            'direction_accuracy': direction_accuracy,  # 总体方向准确率
            'up_accuracy': up_accuracy,               # 上涨预测准确率
            'down_accuracy': down_accuracy,           # 下跌预测准确率
            'max_trend_streak': max_trend_streak,     # 最大连续正确预测次数
            'avg_trend_period': avg_trend_period,     # 平均趋势持续期
            'trend_periods': trend_periods            # 所有趋势持续期列表
        }
    except Exception as e:
        logger.error(f"计算方向性指标时出错: {str(e)}")
        raise


def evaluate_predictions_by_timeframe(model, data, stock_code):
    """
    评估不同时间尺度的预测准确性，返回误差、方向性指标和可视化图
    """
    from prophet import Prophet
    import matplotlib.pyplot as plt
    import io, base64
    # 动态导入 get_plot_url
    import importlib.util
    import sys
    get_plot_url = None
    try:
        if 'app' in sys.modules:
            get_plot_url = sys.modules['app'].get_plot_url
        else:
            spec = importlib.util.spec_from_file_location('app', 'app.py')
            app_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(app_module)
            get_plot_url = app_module.get_plot_url
    except Exception as e:
        get_plot_url = None

    if not isinstance(model, Prophet):
        logger.warning(f"多时间尺度评估仅支持Prophet模型，跳过{stock_code}")
        return {}
    try:
        last_date = data.index[-1]
        timeframes = {
            'short_term': {'name': '未来五天预测', 'plot_title': '未来五天预测值及置信区间', 'delta': relativedelta(days=5), 'description': '未来5天的预测'},
            'medium_term': {'name': '未来一个月预测', 'plot_title': '未来一个月预测值及置信区间', 'delta': relativedelta(months=1), 'description': '未来1个月的预测'},
            'six_month_term': {'name': '未来六个月预测', 'plot_title': '未来六个月预测值及置信区间', 'delta': relativedelta(months=6), 'description': '未来6个月的预测'},
            'long_term': {'name': '未来一年预测', 'plot_title': '未来一年预测值及置信区间', 'delta': relativedelta(years=1), 'description': '未来1年的预测'}
        }
        results = {}
        for timeframe, config in timeframes.items():
            end_date = last_date + config['delta']
            # 生成足够长的future
            periods = (end_date - last_date).days
            future = model.make_future_dataframe(periods=periods, freq='D')
            forecast = model.predict(future)
            # 精确筛选目标区间
            mask = (forecast['ds'] > last_date) & (forecast['ds'] <= end_date)
            forecast_term = forecast[mask]
            ds_list = forecast_term['ds'].dt.strftime('%Y-%m-%d').tolist()
            y_pred = forecast_term['yhat'].values
            # 误差指标（用历史数据最后N天与预测对比）
            n = len(forecast_term)
            if len(data) >= n and n > 0:
                y_true = data['close'].values[-n:]
                y_pred_eval = y_pred[:n]
                rmse = float(np.sqrt(np.mean((y_true - y_pred_eval) ** 2)))
                mae = float(np.mean(np.abs(y_true - y_pred_eval)))
                mape = float(np.mean(np.abs((y_true - y_pred_eval) / y_true)))
            else:
                rmse = mae = mape = None
            # 方向性指标
            if len(data) >= n and n > 0:
                direction_metrics = evaluate_direction_metrics(y_true, y_pred_eval)
            else:
                direction_metrics = {}
            # 可视化
            forecast_plot = None
            if timeframe in ['medium_term', 'six_month_term', 'long_term'] and get_plot_url is not None and n > 0:
                fig = plt.figure(figsize=(14, 7) if timeframe=='long_term' else (10, 5))
                ds_dt = forecast_term['ds']
                plt.plot(ds_dt, y_pred, label='预测值', color='orange', linewidth=2)
                plt.fill_between(ds_dt, forecast_term['yhat_lower'], forecast_term['yhat_upper'], color='orange', alpha=0.2, label='置信区间')
                plt.title(config['plot_title'])
                plt.xlabel('日期')
                plt.ylabel('预测收盘价')
                ax = plt.gca()
                if timeframe == 'medium_term':
                    ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                    plt.xticks(rotation=45)
                if timeframe == 'six_month_term':
                    ax.xaxis.set_major_locator(mdates.MonthLocator())
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                    plt.xticks(rotation=45, fontsize=10)
                if timeframe == 'long_term':
                    ax.xaxis.set_major_locator(mdates.MonthLocator())
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                    plt.xticks(rotation=45, fontsize=10)
                    ax.set_xlim([last_date + relativedelta(days=1), last_date + relativedelta(years=1)])
                plt.legend()
                plt.tight_layout()
                forecast_plot = get_plot_url(fig)
            metrics = {'rmse': rmse, 'mae': mae, 'mape': mape}
            if timeframe == 'short_term':
                results[timeframe] = {
                    'metrics': metrics,
                    'direction_metrics': direction_metrics,
                    'forecast_dates': ds_list,
                    'forecast_values': y_pred.tolist()
                }
            else:
                results[timeframe] = {
                    'metrics': metrics,
                    'direction_metrics': direction_metrics,
                    'forecast_plot': forecast_plot
                }
        return results
    except Exception as e:
        logger.error(f"评估不同时间尺度预测时出错: {str(e)}")
        raise


def train_and_evaluate_models(stock_codes=None):
    """训练和评估所有模型"""
    try:
        if stock_codes is None:
            stock_codes = ["600519.SH"]  # 默认只处理贵州茅台
        
        # 创建保存模型的目录
        os.makedirs('trained_models', exist_ok=True)
        os.makedirs('model_metrics', exist_ok=True)
        
        # 存储所有股票的评估结果
        all_stocks_metrics = {}
        
        for stock_code in stock_codes:
            logger.info(f"\n开始处理 {stock_code} 的模型训练...")
            
            # 为每只股票创建独立的模型目录
            stock_model_dir = f'trained_models/{stock_code.replace(".", "_")}'
            os.makedirs(stock_model_dir, exist_ok=True)
            
            # 加载和划分数据
            logger.info(f"加载和划分 {stock_code} 的数据...")
            train_data, val_data, test_data = load_and_split_data(stock_code)
            full_data = pd.concat([train_data, val_data, test_data])

            # 确保数据不为空
            if train_data.empty or val_data.empty or test_data.empty:
                raise ValueError(f"{stock_code} 的训练集、验证集或测试集为空")

            # 优化Prophet模型参数
            logger.info(f"开始优化 {stock_code} 的Prophet模型参数...")
            prophet_best_params, prophet_param_results = optimize_prophet_parameters(train_data, val_data)
            
            # 保存Prophet参数优化结果
            prophet_params_file = f'model_metrics/prophet_best_params_{stock_code.replace(".", "_")}.json'
            with open(prophet_params_file, 'w') as f:
                json.dump(prophet_best_params, f, indent=4)
            
            # 合并训练集和验证集用于滚动窗口评估
            trainval_data = pd.concat([train_data, val_data])
            
            # 准备滚动窗口
            windows = rolling_window_validation(trainval_data, window_size=1200, step_size=240)
            
            # 存储Prophet模型的评估结果
            prophet_metrics = []
            
            # 使用tqdm创建进度条
            with tqdm(total=len(windows), desc=f"{stock_code} Prophet模型训练进度") as pbar:
                for i, (start_date, end_date) in enumerate(windows):
                    try:
                        window_train_data = trainval_data[(trainval_data.index >= start_date) &
                                                       (trainval_data.index < end_date)]
                        
                        if window_train_data.empty:
                            continue
                            
                        prophet_model = train_prophet_model(window_train_data, prophet_best_params)
                        
                        if i < len(windows) - 1:
                            next_start, next_end = windows[i + 1]
                            val_window = trainval_data[(trainval_data.index >= next_start) &
                                                    (trainval_data.index < next_end)]
                            
                            if not val_window.empty:
                                prophet_val = val_window.reset_index()[['trade_date', 'close']].copy()
                                prophet_val.columns = ['ds', 'y']
                                prophet_forecast = prophet_model.predict(prophet_val[['ds']])
                                window_metrics = evaluate_model(prophet_val['y'], prophet_forecast['yhat'])
                                prophet_metrics.append(window_metrics)
                                
                    except Exception as e:
                        logger.error(f"处理窗口 {i + 1} 时出错: {str(e)}")
                        continue
                        
                    pbar.update(1)
            
            # 计算Prophet模型的平均评估指标
            if prophet_metrics:
                prophet_avg_metrics = {
                    'rmse': np.mean([m['rmse'] for m in prophet_metrics]),
                    'mae': np.mean([m['mae'] for m in prophet_metrics]),
                    'mape': np.mean([m['mape'] for m in prophet_metrics])
                }
            else:
                raise ValueError(f"{stock_code} 没有成功的Prophet模型评估结果")
            
            # 训练其他模型并评估
            arima_model, last_price = train_arima_model(trainval_data)
            arima_forecast_returns = arima_model.forecast(steps=len(test_data))
            arima_forecast_prices = last_price * np.exp(np.cumsum(arima_forecast_returns))
            arima_metrics = evaluate_model(test_data['close'], arima_forecast_prices)
            
            rf_features = prepare_rf_features(trainval_data)
            rf_model = train_rf_model(rf_features, trainval_data['close'])
            test_features = prepare_rf_features(test_data)
            rf_pred = rf_model.predict(test_features)
            rf_metrics = evaluate_model(test_data['close'], rf_pred)
            
            # Prophet模型在测试集上的评估
            prophet_trainval_df = trainval_data.reset_index()[['trade_date', 'close']].copy()
            prophet_trainval_df.columns = ['ds', 'y']
            test_df = test_data.reset_index()[['trade_date', 'close']].copy()
            test_df.columns = ['ds', 'y']
            prophet_test_model = train_prophet_model(trainval_data, prophet_best_params)
            prophet_test_forecast = prophet_test_model.predict(test_df[['ds']])
            prophet_metrics_test = evaluate_model(test_df['y'], prophet_test_forecast['yhat'])
            
            # 计算方向性指标
            prophet_direction = evaluate_direction_metrics(test_df['y'].values, prophet_test_forecast['yhat'].values)
            arima_direction = evaluate_direction_metrics(test_data['close'].values, arima_forecast_prices)
            rf_direction = evaluate_direction_metrics(test_data['close'].values, rf_pred)

            # 保存评估指标
            stock_metrics = {
                'prophet': {
                    'val': prophet_avg_metrics,
                    'test': prophet_metrics_test,
                    'direction_metrics': prophet_direction
                },
                'arima': {
                    'val': arima_metrics,
                    'test': arima_metrics,
                    'direction_metrics': arima_direction
                },
                'rf': {
                    'val': rf_metrics,
                    'test': rf_metrics,
                    'direction_metrics': rf_direction
                }
            }
            
            all_stocks_metrics[stock_code] = stock_metrics
            
            # 保存每只股票的评估指标
            metrics_file = f'model_metrics/metrics_{stock_code.replace(".", "_")}.json'
            with open(metrics_file, 'w') as f:
                json.dump(stock_metrics, f, indent=4)
            
            # 训练最终模型
            logger.info(f"训练 {stock_code} 的最终模型...")
            
            # 1. 最终Prophet模型
            final_prophet_model = train_final_prophet_model(full_data, prophet_best_params)
            joblib.dump(final_prophet_model, f"{stock_model_dir}/prophet_model_final.joblib")
            
            # 2. 最终ARIMA模型
            final_arima_model, final_last_price = train_arima_model(full_data)
            arima_final_data = {
                'model': final_arima_model,
                'last_price': final_last_price
            }
            joblib.dump(arima_final_data, f"{stock_model_dir}/arima_model_final.joblib")
            
            # 3. 最终随机森林模型
            final_rf_features = prepare_rf_features(full_data)
            final_rf_model = train_rf_model(final_rf_features, full_data['close'])
            joblib.dump(final_rf_model, f"{stock_model_dir}/rf_model_final.joblib")
            
            # 保存最后一个交易日期
            last_trade_date = full_data.index[-1]
            with open(f'model_metrics/last_trade_date_{stock_code.replace(".", "_")}.json', 'w') as f:
                json.dump({'last_trade_date': last_trade_date.strftime('%Y-%m-%d')}, f)
            
            # 在训练完最终模型后，添加多时间尺度评估
            logger.info(f"开始评估 {stock_code} 的多时间尺度预测...")
            
            # 对每个模型进行多时间尺度评估
            for model_name, model in stock_metrics.items():
                if model_name != 'prophet':
                    timeframe_results = evaluate_predictions_by_timeframe(
                        model, 
                        full_data,
                        stock_code
                    )
                    logger.info(f"{model_name} 模型的多时间尺度评估完成")
            
            logger.info(f"{stock_code} 的模型训练和评估完成")
        
        # 保存所有股票的对比结果
        comparison_file = 'model_metrics/all_stocks_comparison.json'
        with open(comparison_file, 'w') as f:
            json.dump(all_stocks_metrics, f, indent=4)
        
        return all_stocks_metrics, True, "所有股票模型训练完成"
        
    except Exception as e:
        logger.error(f"模型训练和评估过程出错: {str(e)}")
        raise


def train_final_prophet_model(full_data, params):
    """用全部数据（2013-2024）训练最终Prophet模型"""
    full_df = full_data.reset_index()[['trade_date', 'close']].copy()
    full_df.columns = ['ds', 'y']

    model = Prophet(
        changepoint_prior_scale=params['changepoint_prior_scale'],
        seasonality_prior_scale=params['seasonality_prior_scale'],
        yearly_seasonality=False,  # 禁用默认年度季节性
        weekly_seasonality=False,  # 禁用默认周季节性
        daily_seasonality=params.get('daily_seasonality', False),
        holidays=cn_holidays,
        mcmc_samples=0
    )
    model.add_seasonality(name='custom_yearly', period=240, fourier_order=10)
    model.add_seasonality(name='custom_monthly', period=20, fourier_order=3)
    model.add_seasonality(name='custom_weekly', period=5, fourier_order=3)
    model.add_country_holidays(country_name='CN')
    model.fit(full_df)
    return model


if __name__ == '__main__':
    print("\n" + "=" * 80)
    print(" " * 20 + "开始模型训练和评估过程" + " " * 20)
    print("=" * 80 + "\n")

    start_time = time.time()
    try:
        # 定义要处理的股票列表
        stock_codes = ["600519.SH", "601328.SH"]  # 贵州茅台和交通银行
        
        metrics, is_complete, message = train_and_evaluate_models(stock_codes)
        end_time = time.time()
        training_time = (end_time - start_time) / 60

        if is_complete:
            # 对每只股票分别打印评估结果
            for stock_code in stock_codes:
                print("\n" + "=" * 80)
                print(f"股票 {stock_code} 的模型评估结果")
                print("=" * 80)
                # 仅传入该股票的模型指标以避免 KeyError
                print_completion_message(metrics[stock_code], training_time)
        else:
            print("\n" + "=" * 80)
            print(" " * 20 + "训练未完成！" + " " * 20)
            print("=" * 80)
            print(f"\n错误信息: {message}")
            print("\n请检查日志文件获取详细信息")
            print("=" * 80 + "\n")
            sys.exit(1)

    except Exception as e:
        print("\n" + "=" * 80)
        print(" " * 20 + "训练过程出错！" + " " * 20)
        print("=" * 80)
        print(f"\n错误信息: {str(e)}")
        print("\n请检查日志文件获取详细信息")
        print("=" * 80 + "\n")
        sys.exit(1)
