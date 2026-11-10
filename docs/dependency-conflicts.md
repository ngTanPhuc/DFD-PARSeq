# Dependency conflicts

Ghi lại các xung đột version gặp phải khi setup môi trường, theo quy định
trong README ("do not silently change either side: record the conflict
and the resolution in docs/ and tell the team").

---

## 2026-11-10 — certifi version conflict khi cài PARSeq requirements

**Lệnh gây lỗi:**

uv pip install -r requirements/core.cpu.txt -e .[train,test]


**Lỗi:**

error: No solution found when resolving dependencies
cause: Because there is no version of certifi==2024.2.2 and
you require certifi==2024.2.2, we can conclude that
your requirements are unsatisfiable.


**Nguyên nhân:** `uv` mặc định chỉ tìm package trên index đầu tiên khai
báo có chứa package đó (ở đây là PyTorch CPU wheel index), để tránh tấn
công "dependency confusion". `certifi==2024.2.2` mà PARSeq yêu cầu chỉ
có trên PyPI, không có trên PyTorch index, nên bị coi là "không tồn tại".

**Giải pháp áp dụng:** thêm cờ `--index-strategy unsafe-best-match`:

uv pip install -r requirements/core.cpu.txt -e .[train,test] --index-strategy unsafe-best-match

Lệnh này cho phép uv tìm trên tất cả index thay vì dừng ở index đầu tiên.

**Trạng thái:** đã giải quyết.

**Người ghi:** Tri (role B)

---

## 2026-11-10 — torch và ~35 package khác bị downgrade khi cài PARSeq

**Quan sát:** Sau `uv sync` (cài theo `pyproject.toml` của repo chính),
torch là `2.15.0`. Sau khi cài PARSeq bằng lệnh ở trên, torch bị đổi
thành `2.2.1+cpu`, cùng với nhiều package khác đổi version, ví dụ:

| Package | Trước (repo) | Sau (PARSeq) |
|---|---|---|
| torch | 2.15.0 | 2.2.1+cpu |
| torchvision | 0.29.2 | 0.17.1+cpu |
| numpy | 2.5.4 | 1.26.4 |
| pytorch-lightning | 2.6.6 | 2.2.0.post0 |
| pillow | 13.0.0 | 10.2.0 |
| hydra-core | 1.3.8 | 1.3.2 |
| timm | 1.0.31 | 0.9.16 |

(danh sách đầy đủ ~38 package, xem log cài đặt ngày 10/11/2026 nếu cần)

**Nguyên nhân:** `requirements/core.cpu.txt` của PARSeq pin các version
cũ hơn nhiều so với `pyproject.toml` của repo `DFD-PARSeq`. PARSeq được
public từ 2022 và không cập nhật theo kịp các bản mới của torch/numpy/...

**Trạng thái:** CHƯA GIẢI QUYẾT — xem mục tiếp theo, liên quan trực tiếp.

**Người ghi:** Tri (role B)

---

## 2026-11-10 — `uv run` tự động ghi đè lại môi trường, xóa mất bản cài PARSeq

**Quan sát:** Sau khi cài PARSeq xong (torch ở mức 2.2.1+cpu như trên),
chạy một lệnh bất kỳ qua `uv run` (ví dụ `uv run pytest`) khiến uv tự
phát hiện `.venv` không khớp với `pyproject.toml`/`uv.lock` của repo
chính, và **tự động cài lại toàn bộ ~38 package** theo đúng
`pyproject.toml`. Kết quả: torch quay lại `2.15.0`, coi như PARSeq bị
gỡ cài đặt ngầm mà không có cảnh báo rõ ràng nào ngoài log cài đặt.

**Xác nhận bằng:**

python -c “import torch; print(torch.version)” # -> 2.2.1+cpu (ngay sau khi cài PARSeq)
uv run pytest
python -c “import torch; print(torch.version)” # -> 2.15.0 (sau khi chạy uv run)


**Nguyên nhân gốc:** `uv run <cmd>` luôn đồng bộ `.venv` theo
`pyproject.toml`/`uv.lock` trước khi chạy lệnh — đây là hành vi mặc định
của `uv`, không phải lỗi. Việc cài PARSeq bằng `uv pip install` (ngoài
phạm vi `uv.lock`) không "bền" qua các lần gọi `uv run`.

**Hệ quả:** không thể dùng chung 1 venv duy nhất cho cả core repo
(quản lý bởi `uv`/`pyproject.toml`) và PARSeq (quản lý bởi
`pip`/`requirements/core.cpu.txt`) theo đúng hướng dẫn hiện tại trong
README (bước 3-4 của Setup).

**Trạng thái:** CHƯA GIẢI QUYẾT — cần cả nhóm thống nhất hướng đi trước
khi điền bảng "Pinned versions" và tiếp tục các bước huấn luyện.

**Các hướng đề xuất (chưa chọn):**
1. **Venv riêng cho PARSeq**: tạo virtual environment thứ hai chỉ dùng
   cho PARSeq (`python -m venv` hoặc `uv venv` riêng), không dùng
   `uv run` cho các lệnh liên quan tới PARSeq mà activate venv đó rồi
   gọi `python`/`pytest` trực tiếp. Ưu điểm: không đụng tới
   `pyproject.toml`. Nhược điểm: phải nhớ chuyển đổi venv, dễ nhầm lẫn.
2. **Gộp PARSeq vào `pyproject.toml`** như một dependency group, để
   `uv` quản lý version thống nhất cho toàn bộ repo. Ưu điểm: một venv
   duy nhất. Nhược điểm: phải tự tay giải lại toàn bộ conflict version
   (torch, numpy, pytorch-lightning, ...) giữa yêu cầu của PARSeq và
   các thư viện khác trong repo — có thể PARSeq không chạy được với
   torch/numpy bản mới.
3. **Dùng Kaggle/Colab làm môi trường chính để train** (đã có trong kế
   hoạch nhóm — "Compute: Kaggle/Colab"), và chỉ dùng venv local cho
   việc phát triển code không cần chạy PARSeq trực tiếp (viết
   data/labels/evaluation code không phụ thuộc PARSeq).

**Người ghi:** Tri (role B), 10/11/2026