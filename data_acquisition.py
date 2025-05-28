import tushare as ts
import pandas as pd
import logging
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_stock_data(ts_code="600519.SH", start_date="20130101", end_date="20241231"):
    """
    获取单只股票数据
    
    参数:
        ts_code (str): 股票代码，默认为贵州茅台
        start_date (str): 开始日期，格式：YYYYMMDD
        end_date (str): 结束日期，格式：YYYYMMDD，默认为当前日期
    
    返回:
        pandas.DataFrame: 处理后的股票数据
    """
    try:
        # 设置token
        ts.set_token('546fae42a1d3c5a4ae6ab225967488e125c41fbb24df31f92e763b47')
        pro = ts.pro_api()
        
        # 如果没有指定结束日期，使用当前日期
        if end_date is None:
            end_date = datetime.now().strftime("%Y%m%d")
        
        logger.info(f"开始获取 {ts_code} 从 {start_date} 到 {end_date} 的数据")
        
        # 获取数据
        df = pro.daily(**{
            "ts_code": ts_code,
            "start_date": start_date,
            "end_date": end_date,
        }, fields=[
            'ts_code',          # 股票代码
            'trade_date',       # 交易日期
            'open',            # 开盘价
            'high',            # 最高价
            'low',             # 最低价
            'close',           # 收盘价
            'vol',             # 成交量
            'amount'           # 成交额
        ])
        
        if df.empty:
            raise ValueError(f"未获取到 {ts_code} 的数据")
        
        # 数据验证
        required_columns = ['ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'vol', 'amount']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"数据缺少必要的列: {missing_columns}")
        
        # 检查数据类型
        numeric_columns = ['open', 'high', 'low', 'close', 'vol', 'amount']
        for col in numeric_columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 按日期升序排列
        df = df.sort_values("trade_date", ascending=True)
        
        # 转换日期格式：YYYYMMDD → YYYY-MM-DD
        df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d").dt.strftime("%Y-%m-%d")
        
        # 检查数据质量
        logger.info(f"数据统计信息：")
        logger.info(f"数据条数: {len(df)}")
        logger.info(f"日期范围: {df['trade_date'].min()} 到 {df['trade_date'].max()}")
        logger.info(f"缺失值统计:\n{df.isnull().sum()}")
        
        return df
        
    except Exception as e:
        logger.error(f"获取数据时出错: {str(e)}")
        raise

def get_multiple_stocks_data(stock_codes, start_date="20130101", end_date="20241231"):
    """
    获取多只股票数据
    
    参数:
        stock_codes (list): 股票代码列表，例如 ["600519.SH", "601328.SH"]
        start_date (str): 开始日期，格式：YYYYMMDD
        end_date (str): 结束日期，格式：YYYYMMDD，默认为当前日期
    
    返回:
        dict: 股票代码为key，对应的DataFrame为value的字典
    """
    try:
        results = {}
        for ts_code in stock_codes:
            logger.info(f"开始获取 {ts_code} 的数据...")
            df = get_stock_data(ts_code, start_date, end_date)
            results[ts_code] = df
            
            # 保存单只股票数据
            output_file = f"tushare_daily_{ts_code.replace('.', '_')}.csv"
            df.to_csv(output_file, index=False)
            logger.info(f"{ts_code} 数据已保存到 {output_file}")
        
        return results
        
    except Exception as e:
        logger.error(f"获取多只股票数据时出错: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        # 定义要获取的股票列表（贵州茅台和交通银行）
        stock_codes = ["600519.SH", "601328.SH"]  # 贵州茅台和交通银行
        
        # 获取多只股票数据
        stock_data_dict = get_multiple_stocks_data(stock_codes)
        logger.info("所有股票数据获取完成")
        
    except Exception as e:
        logger.error(f"程序执行出错: {str(e)}")