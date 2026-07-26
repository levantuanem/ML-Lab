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
---
# Bước 2. Tạo Virtual Environment
Tạo môi trường Python riêng cho project để tránh xung đột thư viện giữa các dự án.
```bash
python -m venv .venv
```
Sau khi thực hiện sẽ xuất hiện thư mục:
```text
.venv/
```
---
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
---
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
---
# Bước 5. Cập nhật Pip
Nâng cấp pip lên phiên bản mới nhất.
```bash
python -m pip install --upgrade pip
```
Kiểm tra:
```bash
pip --version
```
---
# Bước 6. Cài đặt các thư viện cần thiết
```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter joblib
```
---
# Bước 7. Lưu danh sách thư viện
Xuất toàn bộ thư viện đang cài đặt vào file `requirements.txt`.
```bash
pip freeze > requirements.txt
```
---
# Bước 8. Tạo `.gitignore`
Tạo file:
```text
.gitignore
```
Các file và thư mục trên sẽ không được Git theo dõi.
---
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
---
# Bước 10. Commit đầu tiên
Thêm toàn bộ file vào vùng staging.
```bash
git add .
```
Tạo commit đầu tiên.
```bash
git commit -m "Initial project structure"
```
---
# Bước 11. Đổi branch mặc định
Đổi branch từ `master` sang `main`.
```bash
git branch -M main
```
Kiểm tra:
```bash
git branch
```
---
# Bước 12. Kết nối GitHub Repository
Thêm repository từ GitHub vào project.
```bash
git remote add origin https://github.com/<username>/<repository>.git
```
Kiểm tra:
```bash
git remote -v
```
---
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
---
# Kiểm tra môi trường
```bash
python --version
where python
pip --version
pip list
git status
```
---

+ FEATURE: THUỘC TÍNH ( Dùng trong tiền xử lý dữ liệu)
    + Numerical ( Số):
        + Integer
        + Float
    + Categorical (Phân loại):
        + Nomical: Định danh
        + Ordinal: Thứ bậc
        + Boolean: Logic

+ CÁC PHƯƠNG THỨC HỌC TRONG MACHINE LEARNING
    + Supervised learning: Dùng dữ liệu có đánh nhãn (Với 1 input thì sẽ có 1 output label tương ứng để huấn luyện mô hình)
        + Classification: Bài toán phân loại: Kết quả là 1 nhóm/ lớp (Vô hạn)
        + Regression: Kết quả là 1 số ( Hữu hạn)
    + UnSupervised learning: Dùng dữ liệu không đánh nhãn ( Unlabel data)
        + Clustering: Bài toán phân nhóm/ phân cụm
        (Là thuật toán để phân chia nhóm dữ liệu ra thành các nhóm nhỏ dự trên sự tương đồng giữa các dữ liệu trong mỗi nhóm con)
        + Association: Bài toán tìm ra 1 quy luật nào đó dự trên dữ liệu cho trước
    + Reinforcement learning: 

+ Supervised Learning: Học có giám sát
     + Biến độc lập là feature, biến phụ thuộc là target
     + Sample, item, record được xem là 1 hàng trong dataset
     + Feature, Attribute, Independence: Cột không phải mang đi dự đoán
     + Label (Target), dependence: Cột cần phải dự đoán
     + Trong 1 sample khi tách ra: Những thành phần liên quan đến feature là: feature vector, ô liên quan đến target được gọi là corresponding label(target) 
     + feature matrix (X) có 1 vector tương ứng label/ target vector (Y) độ dài chính là feature matrix

     + Model training ( Huân luyện mô hình)
     + Đưa feature vector và model để training sau đó đưa ra prediction, lấy prediction đó so sánh với label để tạo ra hàm loss ( hàm loss càng thấp thì model càng tốt). Lấy hàm loss cập nhật lại model
     + Về sau khi mà feature vector ít quá => Cần đưa input vào bằng feature matrix

    + Loss function: Hàm mất mát (Càng bé thì càng tốt)
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
                + huber loss: Nếu sai số nhỏ thì dùng L2, sai số lớn thì dùng L1
    + Phân chia dataset:
        + Training set: Tập này dùng để huấn luyện mô hình ( Loss được tính dựa vào prediction và label: Hàm loss dùng để update mô hình)
        + Validation set: Tập này được dùng để trong quá trình huấn luyện, tại 1 thời điểm nhất định, ví dụ ở cuối 1 Epoch: 1 lần fit toàn bộ dữ liệu vào mô hình để đảm bảo rằng mô hình vẫn đang làm việc tốt (Loss dùng để xem mô hình có đang học tốt hay không, không dùng để update mô hình)
        + Test set: Độ chính xác của mô hình dựa trên test set này 
        + (60/20/20) (70/15/15) (90/5/5) Bộ validation và bộ test thường bằng nhau