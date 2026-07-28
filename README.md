# Project_1
Machine Learning Project
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
Mối quan hệ giữa các variables (features, target)
+ Correlation: Tương quan   
    + Là mức độ 2 hay nhiều variables quan hệ tuyến tính với nhau
    + Hệ số tương quan [-1,1]
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
    + Hệ số tương quan = 0: Thì 2 biến này không có mối quan hệ tuyến tính với nhau
    + Hệ số tương quan chỉ ra mức quan hệ tuyến tính giữa 2 hay nhiều biến với nhau. Nhưng không chỉ ra biến này thay đổi nên làm biến kia thay đổi hoặc ngược lại
    => Giữa target và feature nếu có mối quan hệ tuyến tính với nhau thì lúc đấy sẽ có thiên hướng chọn lựa mô hình tuyến tính. Còn nếu không có mối quan hệ tuyến tính thì sử dụng các mô hình có tính phức tạp hơn, mô hình có thể học được mối quan hệ vượt qua bậc 1, 2, 3, ...,n
```
```bash
Mối quan hệ giữa các features
+ (Multi)collinearity: Cộng tuyến/ Đa cộng tuyến
    + Thể hiện 2 hay nhiều features có quan hệ tuyến tính với nhau
    + Thông thường hệ số tương quan > 0.7 hoặc < -0.7 biểu thị rằng 2 hay nhiều features có hiện tượng (đa) cộng tuyến với nhau
```
```bash
+ Mối quan hệ giữa feature và target thì correlation (hệ số tương quan) có trị tuyệt đối càng cao càng tốt (có nghĩa là càng cần -1, 1 thì càng tốt). Vì khi có 1 hoặc 1 vài feature có mức độ tương quan mạnh với target thì lúc đó ta có thể dễ dàng chọn được các mô hình tuyến tính
+ Mối quan hệ giữa 2 feature ((Multi)collinearity) thì lại không mong hệ số tương quan cao. Tại vì khi hệ số tương quan giữa 2 hay nhiều feature cao thì sẽ có rất nhiều ảnh hưởng tiêu cực đến xây dựng mô hình. Từ đó những vấn đề phát sinh ra do đa cộng tuyến bao gồm: 
    + Thừa thãi
    + Làm 1 feature quan trọng trở thành không quan trọng
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
```bash
Data Visualization
Đồ thị đơn biến
+ Dùng thư viện matplotlib, seaborn
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
Đồ thị đa biến
+ Correlation Matrix Plot (Dùng thư viện seaborn)
    Đánh giá sự tương quan giữa các feature với nhau
    sn.heatmap(df.corr(), annot= True)
    plt.show()
+ Scatter Matrix Plot (Dùng thư viện from pandas.plotting import scatter_matrix)
    scatter_matrix(df)
    plt.show()
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
# 5. Exploratory Data Analysis (EDA)
+ Statistics
+ Visualization
+ Correlation
+ Distribution
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
## Supervised Learning
## UnSupervised Learning
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
```
```bash
Dataset split
Phân chia dataset:
    + Training set: Tập này dùng để huấn luyện mô hình ( Loss được tính dựa vào prediction và label: Hàm loss dùng để update mô hình)
    + Validation set: Tập này được dùng để trong quá trình huấn luyện, tại 1 thời điểm nhất định, ví dụ ở cuối 1 Epoch: 1 lần fit toàn bộ dữ liệu vào mô hình để đảm bảo rằng mô hình vẫn đang làm việc tốt (Loss dùng để xem mô hình có đang học tốt hay không, không dùng để update mô hình)
    + Test set: Độ chính xác của mô hình dựa trên test set này 
    + (60/20/20) (70/15/15) (90/5/5) Bộ validation và bộ test thường bằng nhau
```
```bash
Batch
```
```bash
Epoch
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
```
```bash
Learning rate
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
   