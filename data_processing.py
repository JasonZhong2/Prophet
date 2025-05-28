import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.tsa.stattools import adfuller
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logger = logging.getLogger(__name__)

def analyze_time_series(data):
    """分析时间序列特征"""
    # 创建时间特征
    data['month'] = data.index.month
    data['quarter'] = data.index.quarter
    data['dayofweek'] = data.index.dayofweek
    data['is_monday'] = (data.index.dayofweek == 0).astype(int)
    
    # 计算节前一周（春节和中秋节）
    def is_before_holiday(date):
        # 春节（长假，7天）
        spring_festival = [
            '2013-02-10', '2014-01-31', '2015-02-19', '2016-02-08', '2017-01-28',
            '2018-02-16', '2019-02-05', '2020-01-25', '2021-02-12', '2022-02-01',
            '2023-01-22', '2024-02-10'
        ]
        # 中秋节（短假，3天）
        mid_autumn = [
            '2013-09-19', '2014-09-08', '2015-09-27', '2016-09-15', '2017-10-04',
            '2018-09-24', '2019-09-13', '2020-10-01', '2021-09-21', '2022-09-10',
            '2023-09-29', '2024-09-17'
        ]
        
        date = pd.to_datetime(date)
        
        # 检查春节前
        for holiday in spring_festival:
            holiday = pd.to_datetime(holiday)
            if 0 < (holiday - date).days <= 7:  # 春节前7个交易日
                return 2  # 2表示长假前
        
        # 检查中秋节前
        for holiday in mid_autumn:
            holiday = pd.to_datetime(holiday)
            if 0 < (holiday - date).days <= 3:  # 中秋节前3个交易日
                return 1  # 1表示短假前
        
        return 0  # 0表示非节前
    
    data['is_before_holiday'] = data.index.map(is_before_holiday)
    
    # 计算各因子的平均收益率和标准差
    factors = {
        '是否为周一': data[data['is_monday'] == 1]['log_return'],
        '是否春节前一周': data[data['is_before_holiday'] == 2]['log_return'],
        '是否中秋节前三天': data[data['is_before_holiday'] == 1]['log_return'],
        '季度 = 1': data[data['quarter'] == 1]['log_return'],
        '季度 = 2': data[data['quarter'] == 2]['log_return'],
        '季度 = 3': data[data['quarter'] == 3]['log_return'],
        '季度 = 4': data[data['quarter'] == 4]['log_return']
    }
    
    # 创建结果DataFrame
    results = pd.DataFrame({
        '因子': list(factors.keys()),
        '平均对数收益率': [factor.mean() for factor in factors.values()],
        '标准差': [factor.std() for factor in factors.values()]
    })
    
    # 打印分析结果
    print("\n" + "=" * 80)
    print(" " * 20 + "时间序列特征分析结果" + " " * 20)
    print("=" * 80)
    print(results.to_string(index=False))
    print("=" * 80 + "\n")
    
    return results

def plot_time_series_analysis(data, stock_code=None):
    """绘制时间序列分析图表"""
    plt.figure(figsize=(15, 10))
    
    # 设置图表标题前缀
    title_prefix = f"{stock_code} - " if stock_code else ""
    
    # 1. 原始价格序列
    plt.subplot(2, 2, 1)
    plt.plot(data.index, data['close'])
    plt.title(f'{title_prefix}原始价格序列')
    plt.xticks(rotation=45)
    
    # 2. 对数收益率
    plt.subplot(2, 2, 2)
    plt.plot(data.index, data['log_return'])
    plt.title(f'{title_prefix}对数收益率')
    plt.xticks(rotation=45)
    
    # 3. 季节性差分后的序列
    plt.subplot(2, 2, 3)
    plt.plot(data.index, data['log_return_seasonal_diff'])
    plt.title(f'{title_prefix}季节性差分后的序列')
    plt.xticks(rotation=45)
    
    # 4. 对数收益率的自相关图
    plt.subplot(2, 2, 4)
    pd.plotting.autocorrelation_plot(data['log_return'].dropna())
    plt.title(f'{title_prefix}对数收益率的自相关图')
    
    plt.tight_layout()
    
    # 保存图表
    output_file = f'time_series_analysis_{stock_code.replace(".", "_")}.png' if stock_code else 'time_series_analysis.png'
    plt.savefig(output_file)
    plt.close()

def preprocess_data(data, stock_code=None):
    """
    数据预处理主函数
    
    参数:
        data (pandas.DataFrame): 原始股票数据
        stock_code (str): 股票代码，用于日志记录
    
    返回:
        pandas.DataFrame: 处理后的股票数据
    """
    try:
        if stock_code:
            logger.info(f"开始处理 {stock_code} 的数据...")
        
        # 1. 日期格式标准化和排序
        data['trade_date'] = pd.to_datetime(data['trade_date'])
        data = data.sort_values('trade_date')
        data = data.set_index('trade_date')

        # 2. 处理缺失值
        date_diff = data.index.to_series().diff().dt.days
        mask_short_gap = (date_diff > 1) & (date_diff <= 7)
        if mask_short_gap.any():
            data = data.interpolate(method='linear')
        mask_long_gap = date_diff > 7
        if mask_long_gap.any():
            data = data[~mask_long_gap]

        # 3. 异常值检测和处理
        z_scores = np.abs(stats.zscore(data['close']))
        ma7 = data['close'].rolling(window=7, center=True).mean()
        Q1 = data['close'].quantile(0.25)
        Q3 = data['close'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        for i in range(len(data)):
            if z_scores[i] > 3:
                if i > 0 and i < len(data) - 1:
                    data.iloc[i, data.columns.get_loc('close')] = ma7.iloc[i]
                else:
                    if data['close'].iloc[i] < lower_bound or data['close'].iloc[i] > upper_bound:
                        data.iloc[i, data.columns.get_loc('close')] = data['close'].median()

        # 4. 计算对数收益率
        data['log_return'] = np.log(data['close']).diff().dropna()
        
        # 5. 进行季节性差分（12个月）
        data['log_return_seasonal_diff'] = data['log_return'].diff(12).dropna()
        
        # 6. ADF检验（对季节性差分后的序列）
        adf_result = adfuller(data['log_return_seasonal_diff'].dropna())
        if stock_code:
            print(f"\n{stock_code} ADF检验结果（季节性差分后）:")
            print(f"ADF统计量: {adf_result[0]:.4f}")
            print(f"p值: {adf_result[1]:.4f}")
            print("临界值:")
            for key, value in adf_result[4].items():
                print(f"\t{key}: {value:.4f}")
            if adf_result[1] < 0.05:
                print("结论: 季节性差分后的序列是平稳的 (p值 < 0.05)")
            else:
                print("结论: 季节性差分后的序列不是平稳的 (p值 >= 0.05)")

        # 7. 进行时间序列特征分析
        time_effects_results = analyze_time_series(data)
        
        # 8. 绘制时间序列分析图表
        # NOTE: 不再保存时序分析图到文件夹
        # if stock_code:
        #     plot_time_series_analysis(data, stock_code)

        if data.empty:
            raise ValueError("预处理后的数据为空")
            
        return data
        
    except Exception as e:
        logger.error(f"数据预处理出错: {str(e)}")
        raise

def preprocess_multiple_stocks(stock_data_dict):
    """
    批量处理多只股票数据
    
    参数:
        stock_data_dict (dict): 股票代码为key，对应的DataFrame为value的字典
    
    返回:
        dict: 股票代码为key，处理后的DataFrame为value的字典
    """
    try:
        processed_data_dict = {}
        for stock_code, data in stock_data_dict.items():
            logger.info(f"\n开始处理 {stock_code} 的数据...")
            processed_data = preprocess_data(data, stock_code)
            processed_data_dict[stock_code] = processed_data
            
            # 保存处理后的数据
            output_file = f"processed_data_{stock_code.replace('.', '_')}.csv"
            processed_data.to_csv(output_file)
            logger.info(f"{stock_code} 处理后的数据已保存到 {output_file}")
        
        return processed_data_dict
        
    except Exception as e:
        logger.error(f"批量处理股票数据时出错: {str(e)}")
        raise

if __name__ == '__main__':
    try:
        # 读取多只股票的原始数据
        stock_codes = ["600519.SH", "601328.SH"]  # 贵州茅台和交通银行
        stock_data_dict = {}
        
        for stock_code in stock_codes:
            input_file = f"tushare_daily_{stock_code.replace('.', '_')}.csv"
            stock_data_dict[stock_code] = pd.read_csv(input_file)
        
        # 批量处理数据
        processed_data_dict = preprocess_multiple_stocks(stock_data_dict)
        logger.info("所有股票数据预处理完成")
        
    except Exception as e:
        logger.error(f"程序执行出错: {str(e)}")
