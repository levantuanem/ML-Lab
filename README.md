# Project_1
Machine Learning Project:
DATA
✓ Missing values
✓ Duplicates
✓ Categorical encoding
✓ Outlier
✓ Data leakage

FEATURE
✓ Scaling
✓ Feature selection
✓ Feature importance

MODEL
✓ Logistic Regression
✓ Random Forest
✓ Gradient Boosting
✓ SVM

IMBALANCE
✓ Class weights
✓ RandomOverSampler
✓ SMOTE

VALIDATION
✓ Stratified K-Fold
✓ Cross Validation

OPTIMIZATION
✓ GridSearchCV
✓ RandomizedSearchCV

ROBUSTNESS
✓ Confusion Matrix
✓ Precision
✓ Recall
✓ F1
✓ ROC-AUC
✓ PR-AUC

ENGINEERING
✓ Pipeline
✓ Config
✓ Logging
✓ Model saving
✓ Reproducibility
## Environment
Python 3.13
## Libraries
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn

# 🚀 Project Initialization
---
# Bước 1. Khởi tạo Git Repository
Khởi tạo Git để quản lý phiên bản của project.
```bash
git init
```
Kiểm tra trạng thái repository:
```bash
git status
```
# Bước 2. Tạo Virtual Environment
Tạo môi trường Python riêng cho project để tránh xung đột thư viện giữa các dự án.
```bash
python -m venv .venv
```
Sau khi thực hiện sẽ xuất hiện thư mục:
```text
.venv/
```
# Bước 3. Kích hoạt Virtual Environment
### Windows CMD
```cmd
.venv\Scripts\activate
```
### Windows PowerShell
```powershell
.venv\Scripts\Activate.ps1
```
Khi kích hoạt thành công, Terminal sẽ hiển thị:
```text
(.venv)
```
# Bước 4. Kiểm tra Python Environment
Kiểm tra phiên bản Python:
```bash
python --version
```
Kiểm tra Python đang được sử dụng:
```bash
where python
```
Đường dẫn đầu tiên phải là:
```text
<Project_Path>\.venv\Scripts\python.exe
```
# Bước 5. Cập nhật Pip
Nâng cấp pip lên phiên bản mới nhất.
```bash
python -m pip install --upgrade pip
```
Kiểm tra:
```bash
pip --version
```
# Bước 6. Cài đặt các thư viện cần thiết
```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter joblib
```
# Bước 7. Lưu danh sách thư viện
Xuất toàn bộ thư viện đang cài đặt vào file `requirements.txt`.
```bash
pip freeze > requirements.txt
```
# Bước 8. Tạo `.gitignore`
Tạo file:
```text
.gitignore
```
Các file và thư mục trên sẽ không được Git theo dõi.
# Bước 9. Tạo cấu trúc thư mục
Tạo các thư mục cần thiết cho project.
```cmd
mkdir data
mkdir notebooks
mkdir src
mkdir models
mkdir reports
mkdir data\raw
mkdir data\processed
mkdir reports\figures
```
Cấu trúc project:
```text
Project/
│
├── .git/
├── .venv/
├── .gitignore
├── README.md
├── requirements.txt
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
├── src/
├── models/
└── reports/
    └── figures/
```
# Bước 10. Commit đầu tiên
Thêm toàn bộ file vào vùng staging.
```bash
git add .
```
Tạo commit đầu tiên.
```bash
git commit -m "Initial project structure"
```
# Bước 11. Đổi branch mặc định
Đổi branch từ `master` sang `main`.
```bash
git branch -M main
```
Kiểm tra:
```bash
git branch
```
# Bước 12. Kết nối GitHub Repository
Thêm repository từ GitHub vào project.
```bash
git remote add origin https://github.com/<username>/<repository>.git
```
Kiểm tra:
```bash
git remote -v
```
# Bước 13. Đẩy project lên GitHub
```bash
git push -u origin main
```
Các lần cập nhật sau chỉ cần:
```bash
git add .
git commit -m "Update project"
git push
```
# Kiểm tra môi trường
```bash
python --version
where python
pip --version
pip list
git status
```
---
---
---
---
---
# 1. TỔNG QUAN VỀ MACHINE LEARNING
```bash
AI là gì ?
```
```bash
ML là gì ?
+ Học máy là lĩnh vực nghiên cứu cung cấp cho máy tính khả năng học mà không cần lập trình rõ ràng
```
```bash
DL là gì ?
```
```bash
Thư viện python phổ biến
| Thư viện     | Mục đích                                 |
| ------------ | ---------------------------------------- |
| NumPy        | Tính toán ma trận, mảng số               |
| Pandas       | Xử lý dữ liệu dạng bảng                  |
| Matplotlib   | Vẽ biểu đồ                               |
| Scikit-learn | Các thuật toán ML truyền thống           |
| SciPy        | Tính toán khoa học                       |
| XGBoost      | Gradient Boosting hiệu quả               |
| LightGBM     | Gradient Boosting tối ưu cho dữ liệu lớn |
| CatBoost     | Boosting xử lý tốt dữ liệu phân loại     |
| TensorFlow   | Deep Learning                            |
| PyTorch      | Deep Learning                            |


import pandas as pd
import numpy as np
# pyrefly: ignore [missing-import]
import pingouin as pg # Tính hệ số tương quan Pearson và p-value.
import sys # Tương tác với hệ thống Python (tham số dòng lệnh, đường dẫn, thoát chương trình...).
from scipy.stats import pearsonr, norm # Làm việc với phân phối chuẩn (Normal Distribution).
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold # Chia dữ liệu thành K phần để Cross Validation.
from sklearn.metrics import r2_score # Đánh giá mô hình hồi quy (Regression).
from statsmodels.stats.outliers_influence import variance_inflation_factor # Kiểm tra đa cộng tuyến (Multicollinearity).
from statsmodels.tools.tools import add_constant # Thêm cột hằng số (Intercept) cho mô hình của Statsmodels.
from factor_analyzer.factor_analyzer import calculate_kmo # Kiểm tra dữ liệu có phù hợp để phân tích nhân tố (EFA) hay không.
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity # Kiểm định Bartlett trước khi làm EFA.
from factor_analyzer import FactorAnalyzer # Phân tích nhân tố khám phá (Exploratory Factor Analysis - EFA).
import matplotlib.pyplot as plt

import joblib: Đây là thư viện để lưu và đọc các đối tượng Python, đặc biệt là các mô hình Machine Learning.
    + Sau khi train xong, muốn lưu: joblib.dump(model,"models/logistic_model.pkl")
    -> Sẽ tạo file: models/ logistic_model.pkl
    + Sau này không cần train lại, chỉ cần dùng: 
    model = joblib.load("modelslogistic_model.pkl")
```
```bash
Pipeline Machine Learning 
+ MÔ HÌNH 7 BƯỚC
    + BƯỚC 1: Data collection: Thu thập dữ liệu
    + BƯỚC 2: Statistics: Hiểu dữ liệu thông qua thống kê
    + BƯỚC 3: Data preprocessing:  Tiền xử lý dữ liệu được chia thành 6 bước nhỏ
        + B1: Data cleaning: Làm sạch dữ liệu
        + B2: Dimensionality reduction: Giảm chiều dữ liệu
        + B3: Feature enginneering: Lựa chọn đặc tính
        + B4: Sampling data: Lấy mẫu dữ liệu
            + Probability sampling: Dựa vào xác suất thống kê
                + Simple random sample: Lấy mẫu bằng cách bốc bừa
                + Systematic sample: Lấy mẫu theo quy tắc
                + Statified sample: Lấy mẫu nhưng vẫn giữ được mức phân phối của data
                + Cluster sample: Lấy mẫu theo nhóm
            + Non-Probability sampling:  Không dựa vào xác suất thống kê
                + Convenience sample: 
                + Purposive sample: Lấy mẫu bằng cách tìm kiếm đối tượng khớp 
                + Snowball sample: Lấy mẫu bằng cách 
                + Quota sample: Lấy mẫu bằng cách
        + B5: Data transformation
            + Step 1: Smoothing
            + Step 2: Aggregation
            + Step 3: Discretization
            + Step 4: Attribute construction
            + Step 5: Generalization
            + Step 6: Normalization 
            + + + Type of data Visualization: 
                    + Distribution plot
                    + Box and whisker plot
                    + Violin plot
                    + Line plot
                    + Histogram
                    + Scatter plot
                    + Pie chart
                    + Area plot
                    + Heatmap
                    + Hexbin plot
    + BƯỚC 6: Imbalanced data
    + BƯỚC 4: Data visualization: Trực quan hóa dữ liệu
        + B1: Develop your research question
        + B2: Get or create your data
        + B3: Clean your data
        + B4: Choose a chart type
        + B5: Choose your tool
        + B6: Prepare data
    + BƯỚC 5: Model building
    + BƯỚC 6: Model evaluation 
    + BƯỚC 7: Model deployment
```
---
# 2. DATASETS(DỮ LIỆU)
```bash
Dataset là tập hợp các dữ liệu được sử dụng để huấn luyện, đánh giá và kiểm thử mô hình Machine Learning.
Một dataset thường được biểu diễn dưới dạng bảng (table), trong đó:
    + Mỗi hàng (row) biểu diễn một mẫu dữ liệu (Sample).
    + Mỗi cột (column) biểu diễn một thuộc tính (Feature) hoặc giá trị cần dự đoán (Target/Label).
```
```bash
Sample: 
    + Sample (hay còn gọi là Instance, Record, Observation hoặc Item) là một đối tượng dữ liệu trong dataset, được biểu diễn bởi một hàng.
    + Mỗi sample chứa:
        + Các giá trị của Feature
        + Giá trị của Target (đối với bài toán Supervised Learning)
```
```bash
Feature
    + Feature (hay còn gọi là Attribute, Independent Variable hoặc Input Variable) là các thuộc tính đầu vào dùng để mô hình học và đưa ra dự đoán.
    + Nói cách khác, Feature chính là thông tin mà mô hình được phép sử dụng để dự đoán Target.
```
```bash
Target (Label)
    + Target (hay còn gọi là Label, Dependent Variable hoặc Output) là giá trị mà mô hình cần dự đoán.
    + Trong bài toán Supervised Learning, mỗi sample đều có một Target tương ứng.
```
```bash
Các feature của tất cả sample ghép lại thành Feature Matrix (X).
Các target của tất cả sample ghép lại thành Target Vector (Y).
```
```bash
Balance Data: Làm cân bằng dữ liệu
    + Oversampling (tăng số mẫu lớp thiểu số)
        + Random Oversampling
        + SMOTE
        + ADASYN
            + Ưu điểm:
                + Không mất dữ liệu.
                + Hữu ích khi dữ liệu ít.
            + Nhược điểm:
                + Có thể gây overfitting nếu chỉ sao chép dữ liệu.
    + Undersampling (giảm số mẫu lớp đa số)
        + Ví dụ:
            + Class 0: 950
            + Class 1: 50
            + Giảm Class 0 xuống còn khoảng 50–100 mẫu.
        + Ưu điểm: Huấn luyện nhanh hơn.
        + Nhược điểm: Mất thông tin do loại bỏ dữ liệu
```
---
# 3. Kiểu dữ liệu (Data Types)
```bash
+ FEATURE: THUỘC TÍNH
    + Numerical ( Số):
        + Integer
        + Float
    + Categorical (Phân loại):
        + Nomical: Định danh
        + Ordinal: Thứ bậc
        + Boolean: Logic
```
---
# 4. Data Preprocessing
+ Data Cleaning
+ Missing Values
+ Duplicate Data
+ Outlier
+ Data Transformation
<!-- + Encoding
+ Feature Scaling
+ Feature Selection
+ Feature Extraction
+ Feature Engineering -->
---
# 5. Exploratory Data Analysis (PHÂN TÍCH DỮ LIỆU)
```bash
1. Statistics
df.describe()
df.mean()
df.median()
df.std()
df.var()
df.min()
df.max()
df.quantile()
=> Nhằm hiểu dữ liệu, kiểm tra giá trị bất thường, quan sát độ phân tán
2. Distribution
+ Mục tiêu:
    + Quan sát dữ liệu phân bố như thế nào.
    + Có bị lệch (Skewness) không?
    + Có Outlier không?
    + Có gần phân phối chuẩn (Normal Distribution) không?
+ Đồ thị:
    + Histogram
    + Density (KDE)
    + Box Plot
3. Data Visualization
+ Dùng thư viện matplotlib, seaborn
+ Đồ thị đơn biến
    + Histogram
        df.hist()
        plt.tight_layout()
        plt.show()
    + Density plot
        df.plot(kind='density', subplots= True, layout= (3, 3), sharex= False)
        plt.tight_layout()
        plt.show()
    + Box Plot
        df.plot(kind='box', subplots= True, layout= (3, 3), sharex= False)
        plt.tight_layout()
        plt.show()
+ Đồ thị đa biến
    + Correlation Matrix Plot (Dùng thư viện seaborn)
        Đánh giá sự tương quan giữa các feature với nhau
        sn.heatmap(df.corr(), annot= True)
        plt.show()
    + Scatter Matrix Plot (from pandas.plotting import scatter_matrix)
        scatter_matrix(df)
        plt.show()
4. Mối quan hệ giữa các variables (features, target)
+ Correlation: Tương quan   
    + Là mức độ 2 hay nhiều variables quan hệ tuyến tính với nhau
    + Hệ số tương quan [-1,1]
        + -1 Tương quan âm hoàn hảo
        + +1 Tương quan dương hoàn hảo
        +  0 Không có tương quan tuyến tính
    + Lưu ý:
        + Correlation không chứng minh quan hệ nhân quả (causation).
        + Chỉ phản ánh mức độ biến thiên cùng nhau theo dạng tuyến tính.
    + Tương quan âm (Negative Correlation)
        + Khi X tăng thì Y giảm.
        + Ví dụ:
        + Giá sản phẩm ↑ → Số lượng bán ↓
        + Tốc độ ↑ → Thời gian đi ↓
    + Tương quan dương (Positive Correlation)   
        + Khi X tăng thì Y cũng tăng.
        + Ví dụ:
        + Số giờ học ↑ → Điểm thi ↑
        + Nhiệt độ ↑ → Doanh số kem ↑
=> 
+ Nếu feature và target có tương quan tuyến tính mạnh thì các mô hình tuyến tính (Linear Regression, Logistic Regression...) thường hoạt động tốt và dễ diễn giải hơn.
+ Nếu tương quan thấp có thể học được quan hệ phi tuyến: 
    + Decision Tree
    + Random Forest
    + XGBoost
    + Neural Network
    + SVM (kernel)
    + KNN
Tuy nhiên không thể kết luận chỉ nhìn vào correlation.
5. Mối quan hệ giữa các features
+ (Multi)collinearity: Cộng tuyến/ Đa cộng tuyến
    + Thể hiện 2 hay nhiều features có quan hệ tuyến tính với nhau
    + Thông thường hệ số tương quan > 0.7 hoặc < -0.7 biểu thị rằng 2 hay nhiều features có hiện tượng (đa) cộng tuyến với nhau
+ Mối quan hệ giữa feature và target thì correlation (hệ số tương quan) có trị tuyệt đối càng cao càng tốt (có nghĩa là càng cần -1, 1 thì càng tốt). Vì khi có 1 hoặc 1 vài feature có mức độ tương quan mạnh với target thì lúc đó ta có thể dễ dàng chọn được các mô hình tuyến tính
+ Mối quan hệ giữa 2 feature ((Multi)collinearity) thì lại không mong hệ số tương quan cao. Tại vì khi hệ số tương quan giữa 2 hay nhiều feature cao thì sẽ có rất nhiều ảnh hưởng tiêu cực đến xây dựng mô hình. 
Hai feature chứa gần như cùng một thông tin.
Mô hình sẽ khó biết nên ưu tiên feature nào.
    + Hậu quả
        + Dữ liệu dư thừa
        + Hệ số mô hình không ổn định
        + Khó giải thích mô hình
        + Tăng phương sai của hệ số
        + Làm giảm khả năng diễn giải
    + Đặc biệt ảnh hưởng đến
        + Linear Regression
        + Logistic Regression
    + Ít ảnh hưởng hơn tới
        + Random Forest
        + Decision Tree
        + XGBoost
```
---
# 6. Các phương thức học trong ML
```bash
Supervised Learning
    + Supervised Learning là phương pháp học sử dụng dữ liệu đã được gắn nhãn (Labeled Data).
    + Mỗi sample trong dataset bao gồm:
        + Input (Feature)
        + Output (Target/Label)
    + Mô hình sẽ học mối quan hệ giữa Feature và Target để có thể dự đoán Target cho những dữ liệu mới chưa từng xuất hiện.
UnSupervised Learning
    + Unsupervised Learning là phương pháp học sử dụng dữ liệu không được gắn nhãn (Unlabeled Data).
    + Dataset chỉ chứa Feature, không có Target.
    + Mục tiêu của mô hình là khám phá cấu trúc hoặc quy luật tiềm ẩn trong dữ liệu.
Reinforcement Learning
    + Reinforcement Learning là phương pháp học trong đó một tác nhân (Agent) tương tác với môi trường (Environment) để học cách đưa ra hành động tối ưu.
    + Sau mỗi hành động, môi trường sẽ trả về:
        + Reward (Phần thưởng) nếu hành động tốt.
        + Penalty (Hình phạt) nếu hành động không tốt.
    + Mục tiêu của Agent là tối đa hóa tổng phần thưởng nhận được theo thời gian.
Semi-supervised Learning
```
## Supervised Learning (HỌC CÓ GIÁM SÁT)
+ Đặc điểm chính của học có giám sát là cung cấp cho thuật toán học các ví dụ để học hỏi bao gồm các câu trả lời đúng (input -> outputlabel)
+ Thuật toán học cách chỉ lấy đầu vào input mà không cần nhãn đầu ra và đưa ra dự đoán tương đối chính xác về đầu ra


+ Regression (HỒI QUY): DỰA ĐOÁN NHỮNG GIÁ TRỊ LIÊN TỤC

    + Linear Regression (MÔ HÌNH HỒI QUY TUYẾN TÍNH)
        + Giả định mối quan hệ tuyến tính giữa feature và target
        + Phù hợp với các bộ linear dataset
        + Dự đoán bằng một đường thẳng
        + Mô hình này rất nhạy cảm với các giá trị outliers
        + Simple linear regression
        + Multi linear regression

    + Polynominal Regression (MÔ HÌNH HỒI QUY ĐA THỨC)
        + Dùng khi dữ liệu có mối quan hệ phi tuyến
        + Tạo thêm các đặc trưng đa thức 
        + Degree (bậc đa thức):
            + Degree = 2 (Quadratic)
            + Degree = 3 (Cubic)
            + Degree = n
        + Gegree càng lớn -> Mô hình càng phức tạp
        + Dễ bị overfiting nếu chọn degree quá cao
        + Thực chất vẫn là Linear Regression sau khi biến đổi feature

+ Classification (PHÂN LOẠI)

    + Logistic Regression
        + Được xây dựng dựa trên Linear Regression
        + Thay vì dự đoán trực tiếp giá trị liên tục, đầu ra được đưa qua hàm Sigmoid (Logistic Function)
        + Hàm Sigmoid chuyển giá trị từ (-∞, +∞) về khoảng (0, 1)
        + Giá trị đầu ra được hiểu là xác suất (Probability) thuộc lớp Positive
        + Sử dụng một ngưỡng (Threshold), thường là 0.5, để đưa ra quyết định phân loại
            + Probability ≥ 0.5 → Class 1
            + Probability < 0.5 → Class 0
        + Thích hợp cho bài toán Binary Classification
        + Có thể mở rộng thành Multi-class Classification (One-vs-Rest hoặc Softmax)
        + Ưu điểm:
            + Đơn giản, dễ diễn giải
            + Huấn luyện nhanh
            + Dự đoán xác suất
        + Nhược điểm:
            + Chỉ học tốt khi ranh giới phân lớp gần tuyến tính
            + Nhạy cảm với outliers

    + Support Vector Machine* (SVM - MÁY VECTOR HỖ TRỢ)
        + Thuật toán phân loại bằng cách tìm siêu phẳng (Hyperplane) tối ưu để phân tách các lớp dữ liệu
        + Mục tiêu là tối đa hóa khoảng cách (Margin) giữa Hyperplane và các điểm dữ liệu gần nhất
        + Các điểm gần Hyperplane nhất được gọi là Support Vectors
        + Hoạt động rất tốt trên dữ liệu có số chiều cao (High-dimensional Data)
        + Có thể xử lý dữ liệu không phân tách tuyến tính bằng Kernel Trick
            + Linear Kernel
            + Polynomial Kernel
            + RBF (Gaussian) Kernel
            + Sigmoid Kernel
        + Có thể dùng cho:
            + Classification
            + Regression (SVR - Support Vector Regression)
        + Ưu điểm:
            + Hiệu quả trên tập dữ liệu vừa và nhỏ
            + Khả năng tổng quát hóa tốt
            + Ít bị Overfitting khi chọn tham số phù hợp
        + Nhược điểm:
            + Huấn luyện chậm với tập dữ liệu rất lớn
            + Nhạy cảm với việc lựa chọn Kernel và các tham số (C, Gamma)
            + Khó diễn giải hơn Decision Tree và Logistic Regression

    + K-nearest neibours* (KNN - THUẬT TOÁN K LÁNG GIỀNG GẦN NHẤT)
        + Thuật toán dựa trên khoảng cách giữa các điểm dữ liệu
        + Chọn K điểm dữ liệu gần nhất với điểm cần dự đoán
        + Classification:
            + Dự đoán theo đa số phiếu (Majority Voting)
        + Regression:
            + Dự đoán bằng trung bình giá trị của K láng giềng
        + Giá trị K:
            + K nhỏ → Dễ Overfitting
            + K lớn → Dễ Underfitting
        + Cần chuẩn hóa dữ liệu (Feature Scaling)
        + Các khoảng cách thường dùng:
            + Euclidean Distance
            + Manhattan Distance
            + Minkowski Distance
            
    + Naive Bayes (THUẬT TOÁN BAYES NGÂY THƠ)
        + Ý tưởng
            + Dựa trên Định lý Bayes để tính xác suất một mẫu thuộc từng class.
            + Giả định các feature độc lập có điều kiện với nhau (Conditional Independence).
            + Chọn class có xác suất hậu nghiệm (Posterior Probability) lớn nhất.
        + Nguyên lý hoạt động
            + Tính xác suất tiên nghiệm (Prior Probability) của từng class.
            + Tính xác suất xuất hiện của từng feature trong mỗi class.
            + Áp dụng Định lý Bayes để tính xác suất hậu nghiệm.
            + So sánh xác suất của các class.
            + Chọn class có xác suất cao nhất.
        + Định lý Bayes
            + Posterior Probability
                + Xác suất thuộc class sau khi quan sát dữ liệu.
            + Prior Probability
                + Xác suất ban đầu của từng class.
            + Likelihood
                + Xác suất xuất hiện của feature khi biết class.
            + Evidence
                + Xác suất xuất hiện của dữ liệu quan sát.
        + Giả định Naive (Conditional Independence)
            + Các feature được giả sử độc lập với nhau khi đã biết class.
            + Đây là giả định giúp việc tính toán trở nên đơn giản và nhanh.
            + Trong thực tế giả định này thường không hoàn toàn đúng nhưng mô hình vẫn hoạt động khá tốt.
        + Các loại Naive Bayes
            + Gaussian Naive Bayes
                + Dùng cho dữ liệu liên tục.
                + Giả sử feature tuân theo phân phối chuẩn (Normal Distribution).
            + Multinomial Naive Bayes
                + Dùng cho dữ liệu đếm (Count Data).
                + Thường dùng trong phân loại văn bản.
            + Bernoulli Naive Bayes
                + Dùng cho dữ liệu nhị phân (0/1).
        + Hyperparameters quan trọng
            + var_smoothing (GaussianNB)
            + alpha (MultinomialNB, BernoulliNB)
            + fit_prior
        + Đặc điểm
            + Dựa trên xác suất.
            + Huấn luyện rất nhanh.
            + Dự đoán rất nhanh.
            + Hoạt động tốt với dữ liệu nhiều chiều.
            + Không yêu cầu dữ liệu tuyến tính.
            + Không dễ bị Overfitting.
        + Ưu điểm
            + Đơn giản.
            + Dễ triển khai.
            + Tốc độ huấn luyện nhanh.
            + Hiệu quả với tập dữ liệu nhỏ.
            + Đặc biệt tốt cho phân loại văn bản.
        + Nhược điểm
            + Giả định các feature độc lập thường không đúng trong thực tế.
            + Hiệu quả giảm khi các feature có tương quan mạnh.
            + Không phù hợp với các bài toán có ranh giới phân lớp rất phức tạp.

    + Decision Tree* (CÂY QUYẾT ĐỊNH)
        + Ý tưởng
            + Mô phỏng quá trình ra quyết định bằng cấu trúc cây.
            + Chia dữ liệu thành các tập con theo các điều kiện của feature.
            + Mục tiêu là tạo ra các node có độ thuần khiết (Purity) cao nhất.
            + Quá trình chia được thực hiện theo phương pháp đệ quy (Recursive Partitioning).
        + Cấu trúc của cây
            + Root Node
                + Nút đầu tiên của cây.
                + Chứa toàn bộ dữ liệu huấn luyện.
                + Là nơi bắt đầu tìm feature tốt nhất để phân chia dữ liệu.
            + Internal Node
                + Các nút trung gian.
                + Mỗi node chứa một điều kiện kiểm tra trên một feature.
                + Mỗi lần chia sẽ tạo ra các node con có độ thuần khiết cao hơn.
            + Branch
                + Nhánh nối giữa các node.
                + Đại diện cho kết quả của điều kiện kiểm tra.
            + Leaf Node
                + Nút cuối cùng.
                + Không tiếp tục phân chia.
                + Chứa kết quả dự đoán.
                    + Classification: Nhãn (Class Label)
                    + Regression: Giá trị dự đoán
        + Quá trình xây dựng cây
            + Bắt đầu từ Root Node.
            + Xét toàn bộ feature có trong tập dữ liệu.
            + Thử tất cả các điểm chia (Split Point) có thể trên từng feature.
            + Đánh giá chất lượng của từng phép chia bằng một hàm đo (Split Criterion).
            + Chọn feature và điểm chia tối ưu.
            + Chia dữ liệu thành các node con.
            + Lặp lại quá trình trên từng node con (Recursive Splitting).
            + Dừng khi đạt điều kiện dừng.
        + Tiêu chí đánh giá phép chia (Split Criterion)
            + Classification
                + Gini Index
                    + Đo mức độ không thuần khiết (Impurity) của node.
                    + Giá trị càng nhỏ thì node càng thuần khiết.
                    + Được sử dụng trong thuật toán CART.
                + Entropy
                    + Đo mức độ hỗn loạn của dữ liệu.
                    + Entropy càng nhỏ thì dữ liệu càng thuần.
                + Information Gain
                    + Là lượng Entropy giảm sau khi chia.
                    + Chọn phép chia có Information Gain lớn nhất.
            + Regression
                + Mean Squared Error (MSE)
                + Mean Absolute Error (MAE)
                + Variance Reduction
                + Chọn phép chia làm giảm phương sai nhiều nhất.
        + Điều kiện dừng (Stopping Criteria)
            + Tất cả mẫu trong node thuộc cùng một class.
            + Không còn feature hoặc không còn điểm chia phù hợp.
            + Đạt max_depth.
            + Số lượng mẫu nhỏ hơn min_samples_split.
            + Số lượng mẫu trong node nhỏ hơn min_samples_leaf.
            + Mức giảm Impurity không đủ lớn.
        + Hyperparameters quan trọng
            + criterion
                + gini
                + entropy
                + log_loss
                + squared_error (Regression)
            + max_depth
                + Giới hạn chiều sâu của cây.
            + min_samples_split
                + Số mẫu tối thiểu để tiếp tục chia node.
            + min_samples_leaf
                + Số mẫu tối thiểu tại một Leaf Node.
            + max_features
                + Số lượng feature được xem xét tại mỗi lần chia.
            + max_leaf_nodes
                + Giới hạn số lượng Leaf Node.
            + splitter
                + best
                + random
        + Đặc điểm
            + Không yêu cầu Feature Scaling.
            + Không yêu cầu dữ liệu tuyến tính.
            + Xử lý được dữ liệu số và dữ liệu phân loại.
            + Mô hình hóa được các mối quan hệ phi tuyến.
            + Có khả năng học các tương tác giữa các feature.
            + Có thể đánh giá Feature Importance.
            + Dễ trực quan hóa và diễn giải.
        + Ưu điểm
            + Dễ hiểu và dễ giải thích.
            + Không cần chuẩn hóa dữ liệu.
            + Có thể xử lý Missing Value (ở một số triển khai).
            + Hoạt động tốt với dữ liệu phi tuyến.
            + Huấn luyện tương đối nhanh.
        + Nhược điểm
            + Dễ Overfitting khi cây quá sâu.
            + Không ổn định với dữ liệu huấn luyện.
            + Dễ thiên vị các feature có nhiều giá trị phân chia.
            + Độ chính xác của một cây đơn thường không cao bằng các mô hình Ensemble.
        + Giảm Overfitting
            + Pre-pruning
                + Giới hạn max_depth.
                + Tăng min_samples_split.
                + Tăng min_samples_leaf.
                + Giới hạn max_leaf_nodes.
            + Post-pruning
                + Cost Complexity Pruning (CCP Pruning).
            + Ensemble Learning
                + Random Forest.
                + Gradient Boosting.
                + XGBoost.
                + LightGBM.
                + CatBoost.

    + Random Forest* (RỪNG NGẪU NHIÊN)
        + Ý tưởng
            + Kết hợp nhiều Decision Tree để tạo thành một mô hình mạnh hơn.
            + Mỗi cây được huấn luyện trên một tập dữ liệu ngẫu nhiên (Bootstrap Sampling).
            + Mỗi lần chia node chỉ xét một tập con ngẫu nhiên của các feature.
            + Kết quả cuối cùng được tổng hợp từ tất cả các cây.
        + Quá trình xây dựng mô hình
            + Tạo N tập dữ liệu bằng Bootstrap Sampling (lấy mẫu có hoàn lại).
            + Huấn luyện một Decision Tree trên mỗi tập dữ liệu.
            + Tại mỗi node:
                + Chọn ngẫu nhiên một số feature.
                + Tìm feature tốt nhất trong tập feature được chọn.
                + Chia node theo tiêu chí tối ưu.
            + Lặp lại cho đến khi xây dựng đủ số lượng cây.
        + Quá trình dự đoán
            + Classification
                + Mỗi cây đưa ra một dự đoán.
                + Kết quả cuối cùng được quyết định bằng Majority Voting.
            + Regression
                + Mỗi cây đưa ra một giá trị dự đoán.
                + Kết quả cuối cùng là giá trị trung bình của tất cả các cây.
        + Các kỹ thuật chính
            + Bootstrap Sampling (Bagging)
                + Tạo nhiều tập dữ liệu ngẫu nhiên từ tập huấn luyện.
            + Random Feature Selection
                + Mỗi node chỉ xem xét một số feature ngẫu nhiên.
            + Majority Voting
                + Áp dụng cho Classification.
            + Averaging
                + Áp dụng cho Regression.
        + Hyperparameters quan trọng
            + n_estimators
                + Số lượng Decision Tree.
            + criterion
                + gini
                + entropy
                + squared_error (Regression)
            + max_depth
            + min_samples_split
            + min_samples_leaf
            + max_features
            + bootstrap
            + random_state
        + Đặc điểm
            + Không cần Feature Scaling.
            + Xử lý tốt dữ liệu tuyến tính và phi tuyến.
            + Giảm Overfitting so với Decision Tree.
            + Có thể đánh giá Feature Importance.
            + Hoạt động tốt trên dữ liệu có nhiều feature.
        + Ưu điểm
            + Độ chính xác cao.
            + Ít bị Overfitting.
            + Chịu được nhiễu và Outliers tốt hơn Decision Tree.
            + Có khả năng xử lý dữ liệu lớn.
            + Ít cần tinh chỉnh tham số hơn nhiều thuật toán khác.
        + Nhược điểm
            + Thời gian huấn luyện lâu hơn Decision Tree.
            + Tiêu tốn nhiều bộ nhớ.
            + Khó diễn giải hơn Decision Tree.
            + Mô hình lớn khi số lượng cây nhiều.
## UnSupervised Learning
## Unsupervised Learning (HỌC KHÔNG GIÁM SÁT)
+ Định nghĩa
    + Là phương pháp học máy sử dụng dữ liệu không có nhãn (Unlabeled Data).
    + Mục tiêu là khám phá cấu trúc, quy luật hoặc mối quan hệ tiềm ẩn trong dữ liệu.
    + Mô hình tự tìm ra các nhóm hoặc đặc trưng mà không có đáp án đúng để huấn luyện.
+ Đặc điểm
    + Không có Output Label.
    + Không thể tính trực tiếp Accuracy như Supervised Learning.
    + Thường dùng để khám phá dữ liệu (Data Exploration).
    + Có thể dùng như bước tiền xử lý trước khi huấn luyện mô hình Supervised Learning.
+ Các bài toán chính
    + Clustering (Phân cụm)
        + Gom các mẫu dữ liệu có đặc điểm giống nhau vào cùng một nhóm.
        + Ví dụ:
            + K-Means
            + Hierarchical Clustering
            + DBSCAN
            + Gaussian Mixture Model (GMM)
    + Dimensionality Reduction (Giảm số chiều dữ liệu)
        + Giảm số lượng feature nhưng vẫn giữ lại phần lớn thông tin.
        + Giúp giảm thời gian huấn luyện và trực quan hóa dữ liệu.
        + Ví dụ:
            + PCA
            + t-SNE
            + UMAP
    + Association Rule Learning (Khai phá luật kết hợp)
        + Tìm mối quan hệ giữa các mục dữ liệu.
        + Ví dụ:
            + Apriori
            + FP-Growth
    + Anomaly Detection (Phát hiện bất thường)
        + Phát hiện các mẫu dữ liệu khác biệt so với phần lớn dữ liệu.
        + Ví dụ:
            + Isolation Forest
            + One-Class SVM
            + Local Outlier Factor (LOF)
+ Ưu điểm
    + Không cần dữ liệu gán nhãn.
    + Khám phá được cấu trúc tiềm ẩn trong dữ liệu.
    + Hữu ích trong giai đoạn Exploratory Data Analysis (EDA).
    + Có thể hỗ trợ Feature Engineering.
+ Nhược điểm
    + Khó đánh giá chất lượng mô hình.
    + Kết quả phụ thuộc vào thuật toán và tham số.
    + Không đảm bảo các cụm tìm được có ý nghĩa trong thực tế.
+ Ứng dụng
    + Phân khúc khách hàng.
    + Hệ thống gợi ý.
    + Phát hiện gian lận.
    + Phân tích hành vi người dùng.
    + Giảm chiều dữ liệu trước khi huấn luyện mô hình.
    + Trực quan hóa dữ liệu nhiều chiều.

+ K-Means Clustering (PHÂN CỤM K-TRUNG BÌNH)
    + Thuật toán Unsupervised Learning.
    + Thuộc nhóm Clustering (Phân cụm).
    + Dùng để phân chia dữ liệu thành K cụm dựa trên mức độ tương đồng.
    + Ý tưởng
        + Chia dữ liệu thành K cụm (Cluster).
        + Mỗi cụm được đại diện bởi một tâm cụm (Centroid).
        + Mỗi điểm dữ liệu được gán vào cụm có Centroid gần nhất.
        + Thuật toán liên tục cập nhật Centroid cho đến khi hội tụ.
    + Thành phần
        + Cluster
            + Nhóm các điểm dữ liệu có đặc điểm giống nhau.
        + Centroid
            + Tâm của cụm.
            + Được tính bằng trung bình của tất cả các điểm trong cụm.
        + Distance Metric
            + Dùng để đo khoảng cách giữa điểm dữ liệu và Centroid.
            + Thường sử dụng:
                + Euclidean Distance
    + Quy trình hoạt động
        + Chọn số lượng cụm K.
        + Khởi tạo ngẫu nhiên K Centroid.
        + Tính khoảng cách từ từng điểm dữ liệu đến các Centroid.
        + Gán mỗi điểm vào Centroid gần nhất.
        + Cập nhật Centroid bằng trung bình của các điểm trong cụm.
        + Lặp lại cho đến khi:
            + Centroid không còn thay đổi.
            + Hoặc đạt số lần lặp tối đa.
    + Hàm mục tiêu
        + Giảm tổng khoảng cách bình phương giữa các điểm dữ liệu và Centroid.
        + Mục tiêu là tạo các cụm có độ phân tán nhỏ nhất.
    + Lựa chọn số cụm K
        + Elbow Method
            + Quan sát điểm "khuỷu tay" trên đồ thị.
        + Silhouette Score
            + Đánh giá mức độ phân tách giữa các cụm.
        + Domain Knowledge
            + Chọn K dựa trên kiến thức của bài toán.
    + Hyperparameters quan trọng
        + n_clusters
            + Số lượng cụm.
        + init
            + Phương pháp khởi tạo Centroid.
                + random
                + k-means++
        + n_init
            + Số lần khởi tạo để tìm nghiệm tốt nhất.
        + max_iter
            + Số vòng lặp tối đa.
        + random_state
            + Cố định kết quả khi khởi tạo ngẫu nhiên.
    + Đặc điểm
        + Không cần dữ liệu có nhãn.
        + Hoạt động tốt với dữ liệu có cụm dạng hình cầu.
        + Nhạy cảm với vị trí khởi tạo Centroid.
        + Nhạy cảm với Outliers.
        + Cần Feature Scaling khi các feature có đơn vị khác nhau.
    + Ưu điểm
        + Đơn giản và dễ triển khai.
        + Huấn luyện nhanh.
        + Hoạt động tốt trên tập dữ liệu lớn.
        + Dễ diễn giải kết quả.
    + Nhược điểm
        + Phải xác định trước số cụm K.
        + Khó xử lý cụm có hình dạng phức tạp.
        + Nhạy cảm với Outliers.
        + Có thể hội tụ đến nghiệm cục bộ (Local Optimum).
        + Kết quả phụ thuộc vào Centroid khởi tạo.
    + Ứng dụng
        + Phân khúc khách hàng.
        + Phân nhóm sản phẩm.
        + Phân tích hành vi người dùng.
        + Phân cụm ảnh.
        + Nén ảnh (Image Compression).
        + Tiền xử lý dữ liệu.
## Semi-supervised Learning
## Reinforcement Learning
---
# 7. Thuật toán
```bash
Các Thuật Toán ML phổ biến
+ Regression
    + Linear Regression
    + Polynomial Regression
    + Ridge Regression
    + Lasso Regression
+ Classification
    + Logistic Regression
    + KNN
    + Decision Tree
    + Random Forest
    + SVM
    + Naive Bayes
    + Gradient Boosting
    + XGBoost
    + LightGBM
    + CatBoost
+ Clustering
    + K-Means
    + DBSCAN
    + Hierarchical Clustering
    + Gaussian Mixture Model
+ Dimensionality Reduction
    + PCA
    + t-SNE
    + UMAP
```
---
# 8. Model Training
```bash
Training process
    + Quy trình huấn luyện mô hình
    + Forward Propagation
    + Loss Function
    + Backpropagation
    + Optimizer
    + Cập nhật Weights
    + Lặp lại đến khi hội tụ
```
```bash
Batch
    + Batch Size
    + Mini-batch
    + Batch Gradient Descent
    + Stochastic Gradient Descent (SGD)
    + Mini-batch Gradient Descent
```
```bash
Epoch
    + Epoch là gì
    + Quan hệ giữa Epoch, Batch và Iteration
```
```bash
Loss Function: Hàm mất mát (Càng bé thì càng tốt)
    + Loss được tính toán dựa trên sự khác biệt giữa prediction và label (khác biệt giữa giá trị dự đoán và giá trị thực tế)
    + Đối với bài toán regression: có 2 hàm loss phổ biến: 
        + Loss L1:
            + Least absolute deviations (Độ lệch tuyệt đối tối thiểu)
            + Absolute error (Độ lệch tuyệt đối)
            + Mean of these Absolute etuyệt(Trung bình dộ lệch tuyệt đối)  
        + Loss L2:
            + Least square errors(Sai số bình phương nhỏ nhất)
            + Squared error (Sai số bình phương)
            + Mean of these Squared error ( Trung bình của sai số bình phương)  
        + Điểm khác nhau: 
            + L1: Dự đoán 10 căn nhà sai số 1 triệu sẽ tương đương dự đoán 1 căn nhà sai số 10 triệu (Nếu xây dựng mô hình thật chính xác. càng chính xác càng tốt chấp nhận vài trường hợp đi sai hẵn đi). Muốn dự đoán thật chính xác thì dùng L1
            + L2: Dự đoán 1 căn nhà sai số 10 triệu bằng 100 căn nhà sai số 1 triệu (Dùng loss L2 khi muốn tối thiểu hóa những sai số lớn. Kiểu chấp nhận sai số nhưng sai số phải nhỏ). Muốn sai số nhỏ thì dùng L2
        + Huber loss: Nếu sai số nhỏ thì dùng L2, sai số lớn thì dùng L1
```
```bash
Optimizer
    + Gradient Descent
    + SGD
    + Momentum
    + RMSProp
    + Adam
    + AdamW
```
```bash
Learning rate
    + Learning Rate là gì
    + Learning Rate quá lớn
    + Learning Rate quá nhỏ
    + Learning Rate Scheduler
```
```bash
Dataset Split (Phân chia dữ liệu): Mục đích của việc chia dataset là để đánh giá khả năng tổng quát hóa (generalization) của mô hình trên dữ liệu chưa từng nhìn thấy.
+ Training Set
    + Dùng để huấn luyện (train) mô hình.
    + Mô hình thực hiện:
        + Forward propagation.
        + Tính Loss giữa Prediction và Ground Truth (Label).
        + Backpropagation.
        + Cập nhật trọng số (Weights) bằng Optimizer.
    + Đây là tập duy nhất được dùng để cập nhật mô hình.
+ Validation Set: Dùng để đánh giá mô hình trong quá trình huấn luyện.    
    + Thường được đánh giá:
        + Sau mỗi Epoch.
        + Hoặc sau một số bước (Steps).
    + Mục đích:
        + Theo dõi khả năng học của mô hình.
        + Phát hiện Overfitting hoặc Underfitting.
        + Điều chỉnh Hyperparameters:
            + Learning Rate
            + Batch Size
            + Number of Epochs
            + Optimizer
            + Network Architecture
        + Dùng cho Early Stopping.
    + Validation Loss không được dùng để cập nhật trọng số.
+ Test Set: Chỉ được sử dụng sau khi mô hình đã huấn luyện hoàn tất.
    + Dùng để đánh giá hiệu suất cuối cùng của mô hình trên dữ liệu hoàn toàn mới.
    + Các chỉ số thường dùng:
        + Accuracy
        + Precision
        + Recall
        + F1-score
        + ROC-AUC
        + MSE, RMSE (Regression)
    + Test Set chỉ nên sử dụng một lần để đánh giá cuối cùng, tránh dùng để điều chỉnh mô hình.
+ Tỷ lệ chia dữ liệu
    + Một số tỷ lệ phổ biến:
        60% / 20% / 20%
        70% / 15% / 15%
        80% / 10% / 10%
        90% / 5% / 5% (khi dữ liệu rất lớn)
    + Thông thường:
        + Validation và Test thường có kích thước tương đương nhau.
        + Không có quy tắc bắt buộc; tỷ lệ phụ thuộc vào kích thước dataset.
+ K-Fold Cross Validation là kỹ thuật đánh giá mô hình bằng cách chia dataset thành K phần (K folds) có kích thước gần bằng nhau.
    + Quy trình:
        + Chia dataset thành K tập.
        + Chọn 1 fold làm Validation.
        + K−1 folds còn lại dùng để Train.
        + Huấn luyện và đánh giá mô hình.
        + Lặp lại đến khi mỗi fold đều được sử dụng làm Validation đúng 1 lần.
        + Lấy trung bình kết quả của K lần đánh giá.
+ Khi nào dùng K-Fold?
    + Dataset nhỏ → Nên dùng K-Fold Cross Validation.
    + Dataset lớn → Thường chỉ cần Train/Validation/Test Split vì K-Fold sẽ tốn thời gian mà lợi ích không đáng kể.
    + Thông thường:
        + K = 5 là lựa chọn phổ biến nhất.
        + K = 10 được dùng khi muốn đánh giá kỹ hơn nhưng chi phí tính toán cao hơn.
```
```bash
Hyperparameter Tuning
    + Hyperparameter là gì:
        + Hyperparameter là các tham số được thiết lập trước khi huấn luyện mô hình.
        + Không được học từ dữ liệu.
        + Ảnh hưởng đến quá trình học và hiệu suất của mô hình.
    + Parameter vs Hyperparameter
        + Parameter
            + Được mô hình tự học trong quá trình train.
            + Ví dụ:
                + Weight
                + Bias
        + Hyperparameter
            + Do người dùng thiết lập.
            + Ví dụ:
                + Learning Rate
                + Batch Size
                + Epoch
                + n_estimators
                + max_depth
                + k (KNN)
                + C (SVM)
    + Tại sao cần Hyperparameter Tuning?
        + Mỗi bộ Hyperparameter cho kết quả khác nhau.
        + Giúp tìm bộ tham số tối ưu.
        + Cải thiện Accuracy/F1/AUC...
        + Giảm Overfitting hoặc Underfitting.
    + Search Methods
        + Manual Search
        + Grid Search
        + Random Search
        + Bayesian Optimization 
    + Grid Search
        + Thử tất cả các tổ hợp Hyperparameter.
        + Tìm kết quả tốt nhất.
        + Chính xác nhưng chậm.
    + Random Search
        + Chọn ngẫu nhiên một số tổ hợp Hyperparameter.
        + Nhanh hơn Grid Search.
        + Phù hợp khi không gian tìm kiếm lớn.
    + GridSearchCV
        + Grid Search + K-Fold Cross Validation.
        + Tự động tìm bộ Hyperparameter tốt nhất.
        + Các thuộc tính:
            + best_params_
            + best_score_
            + best_estimator_
    + RandomizedSearchCV
        + Random Search + K-Fold Cross Validation.
        + Chỉ thử một số tổ hợp ngẫu nhiên.
        + Nhanh hơn GridSearchCV.
```
---
# 9. Model Evaluation
```bash
Đánh giá mô hình
+ Classification    
    + Accuracy
    + Precision
    + Recall
    + F1-score
    + ROC-AUC
    + Confusion Matrix
+ Regression
    + MAE
    + MSE
    + RMSE
    + R² Score
```
---
# 10. Model Improvement
```bash
Cross Validation
```
```bash
Hyperparameter Tuning
```
```bash
Regularization
```
```bash
Bias & Variance
```
```bash
Underfitting
```
```bash
Overfitting
```
---
# 11. Model Deployment
+ Save Model
+ Load Model
+ API
+ Inference
---
# 12. Deep Learning
+ ANN
+ CNN
+ RNN
+ LSTM
+ Transformer
+ Transfer Learning
---
   
---
# 13. NLP: Natural Language Processing
```bash
+ Library:
    + nltk
    + textblod
    + spacy
+ 13.1. Classical NLP
    + 13.1.1. Text preprocessing
        + Step 1: Remove punctuations: Loại bỏ dấu câu (.,!?%@)
        + Step 2: Lower text: Đổi thành chữ thường
        + Step 3: Tokenization: Phân chia text thành word/ sentences
        + Step 4: Remove stopwords: you, is, the, an
        + Step 5: 
            + Stemming: (Send = Sent = Sending)
            + Lemmatization: Tốt hơn stemming
        + Step 6: Handling special characters, numbers, URLs
    + 13.1.2. Text representation: là quá trình chuyển văn bản thành dạng số (numeric vector) để Machine Learning Deep Learning có thể xử lý.
        + 1: Bag of Words (BoW)
            + Biểu diễn văn bản dựa trên tần suất xuất hiện của các từ.
            + Không quan tâm đến thứ tự của từ.
            + Nhược điểm:
                + Không hiểu ngữ nghĩa.
                + Không quan tâm thứ tự từ.
                + Vector có thể rất lớn và sparse.
        + 2: N-grams: Biểu diễn văn bản bằng các chuỗi gồm N từ liên tiếp.
            "I love machine learning"
            + Unigram (N = 1): → ["I", "love", "machine", "learning"]
            + Bigram (N = 2): → ["I love", "love machine", "machine learning"]
            + Trigram (N = 3): → ["I love machine", "love machine learning"]
            => N-grams giúp giữ lại một phần thông tin về thứ tự/ngữ cảnh của từ, tốt hơn BoW đơn thuần
        + 3: TF-IDF (Term frequency-Inverse Document Frequency):
            + Mục tiêu: xác định mức độ quan trọng của một từ trong một document so với toàn bộ corpus
            + TF: Term frequency: Tần số xuất hiện của 1 word trong 1 document
            + IDF: Inverse Document Frequency: Mức độ phổ biến/ hiếm của 1 word trong toàn bộ các document
                + Từ xuất hiện trong nhiều documents → IDF thấp.
                + Từ xuất hiện trong ít documents → IDF cao.
            => Từ xuất hiện nhiều trong document nhưng hiếm trong corpus → TF-IDF cao.
            => Từ xuất hiện phổ biến ở hầu hết documents → TF-IDF thấp.
        + 4: One-hot encoding cho từ
            + Mỗi từ được biểu diễn bằng một vector nhị phân.
            + Chỉ có một phần tử = 1, các phần tử còn lại = 0.
            + Nhược điểm:
                + Vector có chiều rất lớn khi vocabulary lớn.
                + Sparse.
                + Không thể hiện mối quan hệ/ngữ nghĩa giữa các từ.
    + 13.1.3. Classical Machine Learning cho NLP
        + Naive Bayes
        + Logistic Regression
        + SVM
        + Decision Tree / Random Forest
    + 13.1.4. Word Embedding
        + Word2Vec
        + CBOW
        + Skip-gram
        + GloVe
        + FastText

+ 13.2. Deep Learning for NLP
    + 13.2.1. Neural Network cơ bản
        + Embedding layer
        + Feed Forward Neural Network
    + 13.2.2. RNN
        + Recurrent Neural Network
        + Vanishing / Exploding Gradient
    + 13.2.3. LSTM
        + Long Short-Term Memory
        + Cell state
        + Forget gate
        + Input gate
        + Output gate
    + 13.2.4. GRU
        + Gated Recurrent Unit
        + So sánh GRU và LSTM
    + 13.2.5. CNN for NLP
        + 1D Convolution
        + Text classification
    + 13.2.6. Attention Mechanism
        + Query
        + Key
        + Value
        + Self-Attention
    + 13.2.7. Transformer
        + Encoder
        + Decoder
        + Positional Encoding
        + Multi-Head Attention
    + 13.2.8. Pre-trained Language Models
        + BERT
        + RoBERTa
        + GPT
        + T5
    + 13.2.9. Fine-tuning
        + Transfer Learning
        + Fine-tuning pretrained models
        + Hugging Face Transformers

+ 13.3. Các bài toán NLP phổ biến
    + Text Classification
    + Sentiment Analysis
    + Spam Detection
    + Named Entity Recognition (NER)
    + Part-of-Speech Tagging (POS)
    + Text Summarization
    + Machine Translation
    + Question Answering
    + Text Generation
    + Information Extraction

Text preprocessing → BoW → TF-IDF → Classical ML → Word2Vec → RNN → LSTM/GRU → Attention → Transformer → BERT/GPT → Fine-tuning
```

```bash
```
---