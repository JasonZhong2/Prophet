# 股票预测系统

## 项目概述
本项目是一个以时序分析Prophet模型为核心，集成ARIMA、随机森林等多模型的股票预测可视化分析系统，涵盖数据获取、数据处理、模型训练、预测分析和Web可视化等功能。系统支持从Tushare获取实时股票数据。

## 主要功能
- 股票数据获取（支持Tushare）
- 数据预处理和特征工程
- 多模型预测（Prophet、ARIMA、随机森林）
- 模型参数优化与评估
- 预测结果可视化与Web展示

## 项目结构

### 1. 数据获取模块
#### `data_acquisition.py`
- **功能**：负责从Tushare API获取股票数据
- **主要特点**：
  - 支持自定义股票代码和日期范围
  - 自动数据验证和格式转换
  - 完整的错误处理和日志记录
- **输出**：生成`tushare_daily_<股票代码>.csv`文件（如`tushare_daily_600519_SH.csv`）


### 2. 数据处理模块
#### `data_processing.py`
- **功能**：对原始数据进行预处理和特征工程
- **主要特点**：
  - 数据清洗和标准化
  - 异常值检测和处理
  - 计算技术指标、节假日特征等
- **输出**：生成`processed_data_<股票代码>.csv`文件

### 3. 模型训练模块
#### `model_training.py`
- **功能**：训练和评估预测模型
- **主要特点**：
  - 支持Prophet、ARIMA、随机森林等多模型
  - 参数优化与交叉验证
  - 训练结果与指标自动保存
- **输出**：
  - 训练好的模型文件（如`trained_models/<股票代码>/prophet_model_final.joblib`等）
  - 评估指标（如`model_metrics/metrics_<股票代码>.json`等）


### 4. Web应用模块
#### `app.py`
- **功能**：提供Web界面展示分析结果
- **主要特点**：
  - 交互式数据展示
  - 多模型预测结果对比
  - 用户友好的界面

## 安装说明

### 环境要求
- Python >= 3.8
- 有效的Tushare API token

### 依赖包列表
本项目依赖如下主要第三方包：
- pandas==1.3.5
- numpy==1.21.6
- matplotlib==3.5.3
- scikit-learn==1.0.2
- prophet==1.1.4
- statsmodels==0.13.2
- tushare==1.2.89
- flask==2.0.3
- scipy==1.7.3
- seaborn==0.12.2
- joblib==1.1.0
- pmdarima==2.0.3
- tqdm==4.67.1
- holidays==0.58
- python-dateutil==2.9.0.post0

### 安装步骤
1. 克隆项目到本地
2. 创建并激活虚拟环境（推荐）
3. 安装依赖：
   ```bash
   python download_dependencies.py
   ```
   或
   ```bash
   pip install -r requirements.txt
   ```
   > **注意：** 若遇到prophet等包安装问题，建议使用国内镜像源或先手动安装依赖。

## 使用说明

### 数据获取
1. 使用Tushare获取数据：
   ```bash
   python data_acquisition.py
   ```
2. 或生成模拟数据（如有generate_data.py）：
   ```bash
   python generate_data.py
   ```

### 数据处理和模型训练
1. 数据预处理：
   ```bash
   python data_processing.py
   ```
2. 训练模型：
   ```bash
   python model_training.py
   ```

### 查看分析结果
1. 命令行分析（如有stock_analysis.py）：
   ```bash
   python stock_analysis.py
   ```
2. Web界面：
   ```bash
   python app.py
   ```

## 数据与模型文件命名规则
- 原始数据：`tushare_daily_<股票代码>.csv`（如`tushare_daily_600519_SH.csv`）
- 处理后数据：`processed_data_<股票代码>.csv`
- 训练好的模型：`trained_models/<股票代码>/prophet_model_final.joblib`等
- 评估指标：`model_metrics/metrics_<股票代码>.json`等

## 依赖管理与自动下载
- `requirements.txt`：列出所有必要的Python包及其版本
- `setup.py`：项目安装配置和元数据
- `download_dependencies.py`：自动下载依赖包到本地`dependencies/wheels`目录，支持离线安装和国内镜像源

## 注意事项
1. 确保Python版本 >= 3.8
2. 需要有效的Tushare API token
3. 建议使用虚拟环境运行项目
4. 注意数据文件的存储位置和权限
5. 首次运行前请确保已安装所有依赖
6. `generate_data.py`和`stock_analysis.py`为可选/扩展模块，主流程不依赖于它们 

---

## 项目完整使用流程（新手快速上手指南）

### 1. 环境准备
- 安装 [Python 3.8+](https://www.python.org/downloads/)
- 注册并获取 [Tushare](https://tushare.pro/register?reg=7) Token（用于获取股票数据）
- （推荐）使用虚拟环境隔离依赖：
  ```bash
  python -m venv venv
  # Windows
  venv\Scripts\activate
  # macOS/Linux
  source venv/bin/activate
  ```

### 2. 安装依赖
- 推荐使用自动下载脚本（支持国内镜像）：
  ```bash
  python download_dependencies.py
  ```
- 或直接安装：
  ```bash
  pip install -r requirements.txt
  ```
- 如遇`prophet`等包安装失败，建议先升级pip，或手动单独安装：
  ```bash
  pip install --upgrade pip
  pip install prophet -i https://pypi.tuna.tsinghua.edu.cn/simple
  ```

### 3. 配置Tushare Token
- 在`data_acquisition.py`中找到`ts.set_token('...')`，将引号内内容替换为你自己的Tushare Token。

### 4. 获取股票数据
- 运行：
  ```bash
  python data_acquisition.py
  ```
- 程序会自动下载指定股票的历史数据，生成如`tushare_daily_600519_SH.csv`等文件。
- 如需模拟数据，可自行实现`generate_data.py`。

### 5. 数据预处理
- 运行：
  ```bash
  python data_processing.py
  ```
- 会自动读取原始数据，完成清洗、特征工程，生成`processed_data_<股票代码>.csv`。

### 6. 训练与评估模型
- 运行：
  ```bash
  python model_training.py
  ```
- 会自动训练Prophet、ARIMA、随机森林等模型，保存模型文件和评估指标。

### 7. 启动Web可视化系统
- 运行：
  ```bash
  python app.py
  ```
- 打开浏览器访问 http://127.0.0.1:5000/ ，即可交互式查看预测结果和分析。

### 8. （可选）命令行分析
- 如有`stock_analysis.py`，可运行：
  ```bash
  python stock_analysis.py
  ```

### 9. 常见问题与建议
- **依赖安装失败**：优先升级pip，使用国内镜像，或手动安装单个包。
- **Tushare Token无效**：请确认Token填写正确且未过期。
- **数据文件未生成**：请检查数据获取步骤是否成功，或手动检查API调用。
- **模型训练慢/内存占用高**：可减少股票数量或缩短数据区间。
- **Web界面无法访问**：确认`app.py`已正常运行，端口未被占用。

---

如有更多问题，欢迎提交Issue或联系开发者。 