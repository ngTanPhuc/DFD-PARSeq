## 0. Command chạy toàn bộ data audit và splits
```bash
chmod +x scripts/a_data_eval.sh
./a_data_eval.sh
```

## 1. Tổng quan dữ liệu raw (VinText)

* **Quy mô & Cấu trúc thư mục giải nén** (`data/raw/vintext/vietnamese/` — tổng 2.000 ảnh, 56.084 vùng chữ):
* `train_images/`: 1.200 ảnh (`im0001.jpg` … `im1200.jpg`).
* `test_image/`: 300 ảnh (`im1201.jpg` … `im1500.jpg`).
* `unseen_test_images/`: 500 ảnh (`im1501.jpg` … `im2000.jpg`).
* `labels/`: 2.000 file `gt_<n>.txt` (tên file ảnh đệm 4 chữ số như `im0461.jpg`, nhưng file nhãn không đệm số như `gt_461.txt`; ID chuẩn của ảnh được quy ước là `im0461`).
* `general_dict.txt` & `vn_dictionary.txt`: Từ điển đi kèm bộ dữ liệu gốc (**Ko dùng cho thí nghiệm E6** tại nó chứa từ vựng của tập test).
* **Định dạng nhãn**: Mã hóa UTF-8 (không BOM), mỗi dòng là một vùng chữ theo cấu trúc `x1,y1,x2,y2,x3,y3,x4,y4,TRANSCRIPT`. Vì bản thân `TRANSCRIPT` có thể chứa dấu phẩy (có 756 dòng gồm 10 trường phân tách bởi dấu phẩy), parser lấy 8 trường đầu làm tọa độ và nối toàn bộ phần còn lại làm chuỗi nhãn.

**Command tải và giải nén dữ liệu raw:**

```bash
uv sync --group data
mkdir -p data/raw
uv run gdown 1UUQhNvzgpZy7zXBFQp0Qox-BBjunZ0ml -O data/raw/vintext_original.zip
unzip data/raw/vintext_original.zip -d data/raw/vintext

```
Toàn bộ quá trình tui làm được liệt kê theo các phần dưới đây:

1. Audit dữ liệu (EDA)
2. Xác định và phân chia các split: train, val, test
3. Crop hình và chia vào các split (lưu ở dạng .mdb)
4. Viết code cho phần 
---

## 2. Phần Audit dữ liệu và các file output

Script `scripts/audit_dataset.py` chạy trực tiếp từ thư mục đã giải nén (hoặc file .zip) để kiểm tra tính toàn vẹn và rút trích thống kê tổng hợp.

* **Các phát hiện & Quyết định kỹ thuật từ Audit**:

* **Lọc vùng nhiễu**: Có 13.049/56.084 vùng không đọc được gồm `###` (12.984), `######` (2), `##` (4), `#` (4) và nhãn rỗng (55). Hàm `is_usable_transcript()` loại bỏ toàn bộ nhóm này ở mọi giai đoạn, giữ lại **43.035 vùng hợp lệ**.

* **Chuẩn hóa Unicode & Chữ hoa/thường trên tập train**: 100% nhãn gốc đã ở dạng **NFC** (0 nhãn NFD/trộn). Phân bố chữ hoa chiếm 67,9% nên toàn bộ hệ thống giữ nguyên **case-sensitive** (phân biệt hoa/thường).

* **Quy tắc bỏ dấu (`hòa` vs `hoà`)**(kiểu old `hòa` / kiểu new `hoà`): Tập train lẫn lộn cả hai kiểu (371 từ kiểu new `hoà, khoẻ, thuỷ` vs 50 từ kiểu old `hòa, khỏe, thủy`). Tui chốt lấy **kiểu new làm chuẩn**: nhãn tập `train` được chuẩn hóa sang kiểu new khi tạo dữ liệu huấn luyện, còn nhãn `validation` và `test` giữ nguyên như gốc (chỉ chuẩn hóa khi tính metric DER).

* **Mất cân bằng lớp (Train-only)**: Ở cấp ký tự, 86,63% vị trí không có thanh điệu và 88,17% không có dấu phụ (`^`, `˘`, `̛`, `đ`) — thông số quan trọng để Person C thiết kế hàm loss cho mô hình 3 heads.

* **Bộ từ vựng 228 ký tự**: Hợp của cả 3 tập tạo thành đúng **228 ký tự** (tập train chỉ có 226 ký tự, thiếu `ẽ` chỉ xuất hiện ở validation và `ỵ` chỉ xuất hiện ở test). Kiểm tra độ phủ đạt 0 ký tự ngoài từ điển trên cả 3 tập.


| File Output | Đường dẫn | Trạng thái Git | Nội dung & Mục đích sử dụng |
| --- | --- | --- | --- |
| Báo cáo Audit JSON | [data/interim/audit/report.json](../data/interim/audit/report.json) | Ignored | Chứa toàn bộ số đếm, phân phối chiều dài từ, tần suất dấu/thanh điệu (không chứa text nhãn). |
| Tài liệu Audit & Quyết định | [docs/data-audit.md](../docs/data-audit.md) | Tracked | Ghi chép chi tiết số liệu đo được, các quyết định đóng băng quy ước và giới hạn dữ liệu. |
| Bộ ký tự chuẩn (228 chars) | [configs/charset/baseline.txt](../configs/charset/baseline.txt) | Tracked | Danh sách 228 ký tự hợp lệ (mỗi dòng 1 ký tự NFC, không comment vì `#` và dấu cách là ký tự nhãn). |
| Module quy tắc chính tả | [src/dfstr/labels/orthography.py](../src/dfstr/labels/orthography.py]) | Tracked | Chứa hàm `tone_normalize()` chuyển dấu thanh sang kiểu mới (bỏ qua âm đầu `qu` như `quà`). |
| Module đọc & lọc nhãn | [src/dfstr/data/annotations.py](../src/dfstr/data/annotations.py) | Tracked | Định nghĩa chuẩn `format_image_id()` và bộ lọc `is_usable_transcript()` dùng chung toàn repo. |

**Command chạy Audit và tạo Charset:**

```bash
# 1. Chạy audit từ folder đã unzip
uv run python scripts/audit_dataset.py --from_folder data/raw/vintext/vietnamese --out data/interim/audit

# 2. Tạo file charset 228 ký tự và kiểm tra độ phủ
uv run python scripts/make_charset.py --from_folder data/raw/vintext/vietnamese --out configs/charset/baseline.txt
uv run python scripts/audit_dataset.py --from_folder data/raw/vintext/vietnamese --charset configs/charset/baseline.txt --out data/interim/audit

```

---

## 3. Phần Split dữ liệu

* **Cách chia & Cơ sở chia**:

* Không xáo trộn ngẫu nhiên và **ko chia ở cấp độ từ** để chống data leakage (mọi từ cắt ra từ cùng một ảnh gốc phải nằm chung một tập).

* Dữ liệu được ánh xạ trực tiếp 1-1 từ 3 thư mục ảnh rời nhau sẵn có trong bản phát hành VinText. Vì VinText gốc không có thư mục validation riêng, nhóm quy ước dùng `test_image` (300 ảnh) làm tập `validation` và `unseen_test_images` (500 ảnh) làm tập `test`.


* **Kết quả sau khi chia**:

| Tập dữ liệu (Split) | Thư mục gốc VinText | Số ảnh | Tổng vùng chữ | Vùng hợp lệ | Vùng nhiễu bị loại | File Manifest lưu trữ |
| --- | --- | --- | --- | --- | --- | --- |
| **train** | `train_images` | 1.200 | 35.094 | 25.766 | 9.328 | [data/splits/train.txt](../data/splits/train.txt) |
| **validation** | `test_image` | 300 | 8.737 | 7.201 | 1.536 | [data/splits/validation.txt](../data/splits/validation.txt) |
| **test** | `unseen_test_images` | 500 | 12.253 | 10.068 | 2.185 | [data/splits/test.txt](../data/splits/test.txt) |
| **Tổng cộng** | - | **2.000** | **56.084** | **43.035** | **13.049** | [data/splits/provenance.json](../data/splits/provenance.json) |

**Command tạo Split manifests và chạy unit test kiểm tra tính rời nhau:**

```bash
uv run python scripts/make_splits.py --from_zip data/raw/vintext_original.zip --audit-report data/interim/audit/report.json
uv run pytest tests/test_splits.py

```

---

## 4. Crop generation và lưu trữ LMDB (`.mdb`)

* **Quy trình cắt ảnh (`scripts/make_crops.py`)**: Chạy 2 pha tuần tự trên CPU (`geometry` tính toán hình học trước, `write` giải mã ảnh và ghi dữ liệu sau):

1. **Sắp xếp đỉnh & Kiểm tra hình học**: Sắp xếp 4 tọa độ đa giác theo chiều kim đồng hồ bắt đầu từ góc trên-trái. Kẹp (*clamp*) các đỉnh tràn viền vào mép ảnh (giữ lại được 142 vùng sát viền). Loại bỏ các đa giác suy biến: trong 43.035 vùng hợp lệ có **54 vùng lỗi hình học bị loại** (28 trùng đỉnh, 23 đa giác lõm/tự cắt, 3 nằm ngoài khung ảnh).

2. **Biến đổi phối cảnh ở độ phân giải gốc**: Dùng phép biến đổi phối cảnh (*perspective warp*, nội suy cubic) ra kích thước tự nhiên của từ (`width` = cạnh ngang dài nhất, `height` = cạnh dọc dài nhất), lưu PNG không mất dữ liệu. Không thêm padding và **giữ nguyên chiều thẳng đứng** với chữ dọc (`height > width`, có 7.550 ảnh dọc) theo đúng thống nhất với Trí (B).

3. **Xử lý nhãn khi đóng gói**: Nhãn tập `train` được chuẩn hóa NFC + `tone_normalize()` (dấu thanh kiểu new); nhãn `validation` và `test` giữ nguyên NFC như gốc.

4. **Kết quả đầu ra**: Tạo được **42.981 ảnh crop** (`train`: 25.723 | `validation`: 7.193 | `test`: 10.065), ghi metadata hình học (không chứa text nhãn) ra [data/interim/crops/manifest.csv](../data/interim/crops/manifest.csv) và đóng gói vào 3 kho `data/lmdb/{train,validation,test}/data.mdb` (tổng 665 MiB).


* **Tại sao dùng `.mdb` (LMDB) và cách sử dụng**:
* **Lý do**: Đây là định dạng chuẩn mà bộ khung PARSeq gốc sử dụng. Gom hàng chục nghìn file ảnh PNG nhỏ vào 1 file nhị phân duy nhất cho mỗi tập giúp đọc dữ liệu qua memory-mapping cực nhanh, tránh nghẽn cổ chai I/O trên Kaggle/Colab và đảm bảo tính nguyên tử (transaction) khi ghi.

* **Cách dùng**: DataLoader mở kho bằng thư viện `lmdb` và truy xuất theo bộ khóa đánh số từ 1 (`1-based`) của PARSeq: `num-samples` (tổng số mẫu), `image-%09d` (bytes ảnh PNG gốc), và `label-%09d` (chuỗi nhãn UTF-8). Việc resize ảnh về `32×128` (E0) hoặc độ phân giải cao hơn (E1/E3) được thực hiện động trong DataLoader lúc huấn luyện.


**Command chạy thử (dry-run) và chạy sinh toàn bộ ảnh crop + LMDB:**

```bash
# Chạy sinh toàn bộ 42,981 crops vào data/lmdb/ và data/interim/crops/manifest.csv
uv run python scripts/make_crops.py --from_folder data/raw/vintext/vietnamese

```

---

## 5. Evaluation và Metrics

Hệ thống đánh giá trong `src/dfstr/evaluation/`: hàm chấm điểm chỉ nhận vào chuỗi Unicode đã giải mã (`crop_id -> pred_str`) và tự phân tích thành phần ký tự thông qua thư viện chuẩn `unicodedata` của Python (tránh rủi ro một lỗi lập trình làm sai đồng thời cả nhãn huấn luyện lẫn điểm số).

### Quy trình chuẩn hóa & Bóc tách thành phần

1. **Chuẩn hóa NFC & Giữ nguyên hoa/thường**: Cả chuỗi dự đoán (`pred`) và nhãn thật (`gt`) đều được đưa về NFC và cos case-sensitive.

2. **Định nghĩa chữ cái gốc (`base_letters(w)`)**: Chuyển chuỗi sang NFD $\rightarrow$ xóa toàn bộ dấu tổ hợp (*combining marks*) $\rightarrow$ ánh xạ tường minh `đ -> d` và `Đ -> D`. Một từ được coi là **đúng chữ cái gốc (base-correct)** khi `base_letters(pred) == base_letters(gt)`.

3. **Chuẩn hóa vị trí dấu thanh (`tone_normalize()`)**: Áp dụng lên **cả `pred` và `gt**` *chỉ* khi tính nhóm metric về dấu (`DER`, `UDER`, `Intersection DER`, độ chính xác từng thành phần và phân loại lỗi) để không phạt oan các từ viết đúng nhưng khác quy tắc đặt dấu `hòa`/`hoà`. Ngược lại, `Word Accuracy` và `1-NED` tính trên chuỗi NFC nguyên bản (nghiêm ngặt).


### Công thức các thước đo

Gọi $N$ là tổng số từ đánh giá, $A$ là số từ đúng chữ cái gốc (`base-correct`), $E$ là số từ khớp chính xác tuyệt đối (`exact` trên chuỗi nghiêm ngặt), và $K$ là số từ đã đúng chữ cái gốc nhưng vẫn sai dấu sau khi qua `tone_normalize()`:

- **Base letters (Chữ cái gốc)**: Là từ sau khi đã bỏ toàn bộ dấu (sắc, huyền, hỏi, ngã, nặng) và biến `đ` thành `d`. (Ví dụ: "hòa" -> "hoa", "đẹp" -> "dep").
- **N**: Tổng số từ trong dataset.
- **Base-correct (A)**: Số từ mà mô hình đọc đúng phần chữ gốc (không quan tâm dấu).
- **Exact (E)**: Số từ mà mô hình đọc đúng 100% (cả chữ gốc lẫn dấu).
- **Tone-error (K)**: Số từ mô hình đọc đúng chữ gốc nhưng **SAI DẤU** (Tone-error = Base-correct - Exact).

---

* **DER (Diacritic Error Rate — Thước đo chính)**:

$$\text{DER} = \frac{K}{A}$$

Đo tỷ lệ lỗi dấu trên các từ đã nhận dạng đúng khung chữ cái gốc.


* **UDER (Unconditional DER)**:

$$\text{UDER} = \frac{K}{N}$$

Khắc phục nhược điểm mẫu số biến thiên của DER (tránh trường hợp một mô hình đọc sai nhiều chữ cái gốc nên chỉ được chấm DER trên tập con các từ dễ).


* **DER trên tập giao (`der_on_intersection`)**: Lấy giao của tập các từ `base-correct` trên **tất cả** mô hình đem so sánh, sau đó tính DER của từng mô hình trên đúng tập giao này để đảm bảo so sánh công bằng tuyệt đối.


* **Word Accuracy & 1-NED**: $\text{WordAcc} = \frac{E}{N}$ và $\text{1-NED} = 1 - \frac{\text{Levenshtein}(pred, gt)}{\max(\vert{}pred\vert{}, \vert{}gt\vert{})}$ (tính trên chuỗi NFC nghiêm ngặt).


* **Modifier Accuracy & Tone Accuracy**: Tỷ lệ từ khớp toàn bộ dấu phụ (hoặc thanh điệu) sau chuẩn hóa dấu, tính trên tập $A$ từ đúng chữ cái gốc.


* **Tỷ trọng lỗi dấu phục vụ Decision Gate**: $\text{Gate Share} = \frac{K}{N - E}$ (tỷ lệ lỗi thuần dấu trên tổng số từ đoán sai của E0; phải đạt $\ge 20\%$ trên tập validation để đi tiếp).


### Phân loại 6 nhóm lỗi dấu loại trừ lẫn nhau (`src/dfstr/evaluation/errors.py`)

Chỉ áp dụng trên $K$ từ đã đúng chữ cái gốc nhưng sai dấu (sau `tone_normalize()`), đảm bảo tổng số lỗi của 6 nhóm luôn bằng đúng $K$:

1. `missing_tone`: Sai tại duy nhất 1 vị trí ký tự và chỉ thiếu dấu thanh.

2. `wrong_tone`: Sai tại duy nhất 1 vị trí ký tự và bị nhầm sang dấu thanh khác.

3. `extra_tone`: Sai tại duy nhất 1 vị trí ký tự và bị thừa dấu thanh.

4. `wrong_modifier`: Sai tại duy nhất 1 vị trí ký tự và chỉ sai dấu phụ (`^`, `˘`, `̛` hoặc thiếu/thừa dấu phụ).

5. `d_stroke` (`d/đ confusion`): Sai tại duy nhất 1 vị trí ký tự và do nhầm lẫn giữa `d` và `đ`.

6. `combined`: Mọi trường hợp còn lại (sai từ 2 vị trí ký tự trở lên trong từ, hoặc sai đồng thời cả dấu phụ lẫn thanh điệu tại cùng 1 ký tự).


**Command chạy kiểm thử bộ metric và kiểm tra rò rỉ dữ liệu:**

```bash
uv run pytest tests/test_metrics.py tests/test_error_categories.py
uv run python scripts/check_leakage.py --crops data/interim/crops/manifest.csv
uv run pytest tests/test_leakage.py

```

---

## 6. Freeze dữ liệu

* **Những gì ĐÃ FREEZE ([<u>*Frozen*</u>](https://www.youtube.com/watch?v=YVVTZgwYwVo) từ 22/11/2026)**:

* Toàn bộ file danh sách chia tập (`data/splits/train.txt`, `validation.txt`, `test.txt`, `provenance.json`) đã được chốt và lưu mã băm SHA-256 trong [docs/data-audit.md](../docs/data-audit.md) và [docs/freeze.md](../docs/freeze.md).

* Bộ từ vựng 228 ký tự ([configs/charset/baseline.txt](../configs/charset/baseline.txt)).

* Bộ 42.981 ảnh crop trong `data/lmdb/{train,validation,test}` và [data/interim/crops/manifest.csv](../data/interim/crops/manifest.csv).

* Các quy ước chuẩn hóa nhãn (NFC, case-sensitive, `tone_normalize()`) và toàn bộ công thức tính metric / phân loại lỗi trong `src/dfstr/evaluation/`.


**Command kiểm tra tính toàn vẹn SHA-256 của dữ liệu đã đóng băng trước khi train/eval:**

```bash
sha256sum data/splits/train.txt data/splits/validation.txt data/splits/test.txt data/splits/provenance.json configs/charset/baseline.txt
uv run python scripts/check_leakage.py --crops data/interim/crops/manifest.csv
uv run pytest

```