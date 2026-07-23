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

