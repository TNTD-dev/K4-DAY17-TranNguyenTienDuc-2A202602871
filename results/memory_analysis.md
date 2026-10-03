# Phân tích Memory Systems for AI Agent — Day 17

Bài làm của Trần Nguyễn Tiến Đức — MSSV 2A202602871. Phân tích này trả lời bốn
câu hỏi của Bước 8 trong `Guide.md`, tương ứng mục 10 của tài liệu hướng dẫn chi tiết.
Số liệu gốc: [benchmark_results.md](benchmark_results.md),
[benchmark_results.json](benchmark_results.json). Bonus được kiểm chứng riêng tại
[conflict_results.json](conflict_results.json).

## Điều kiện đo và ba lớp memory

Hai agent dùng cùng dataset, thứ tự lượt, câu hỏi và cách đặt thread ID. Recall
được hỏi trong thread mới sau mỗi hội thoại, trước khi hội thoại sau thay fact.
Standard có 10 hội thoại, 101 lượt chat và 14 câu recall; Stress có 16 lượt chat
và 3 câu recall. Mỗi lần đo dùng agent mới và profile benchmark sạch; hai lần
chạy trong process độc lập đã cho output giống nhau.

Baseline giữ toàn bộ message trong RAM theo thread, không ghi profile. Advanced
kết hợp message gần nhất theo thread, `User.md` bền vững theo user và summary
của lịch sử đã compact. Persistent memory giúp nhớ qua thread/process; compact
memory giảm lịch sử phải đưa vào prompt. Summary không thay thế profile.

Ngưỡng compact là 1000 token, giữ 4 message mới nhất. Chạy offline, không gọi
model/judge bên ngoài. Token được ước lượng từ số ký tự; tổng gồm chat và recall.
Recall mỗi câu là 0 khi không khớp fact, 0,5 khi khớp một phần, 1 khi khớp đủ.
Quality là tỷ lệ chuỗi fact kỳ vọng khớp, lấy trung bình theo câu, không đánh giá
độ trôi chảy hay suy luận. Hai điểm này có chung cơ sở substring nên không phải
hai bằng chứng chất lượng độc lập.

## 1. Vì sao Advanced recall tốt hơn?

| Bộ | Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
|---|---|---:|---:|---:|---:|---:|---:|
| Standard | Baseline | 1834 | 18433 | 0% | 0,000 | 0 | 0 |
| Standard | Advanced | 1523 | 28182 | 100% | 1,000 | 316 | 0 |
| Stress | Baseline | 826 | 27257 | 0% | 0,000 | 0 | 0 |
| Stress | Advanced | 822 | 12617 | 100% | 1,000 | 238 | 4 |

Advanced khớp đủ fact ở cả 17 câu recall; Baseline không khớp fact kỳ vọng trong
thread mới. `extract_profile_updates()` lấy assertion của user,
`_prepare_turn()` cập nhật `User.md`, rồi `_offline_response()` đọc profile vào
ngữ cảnh của thread mới. Baseline chỉ còn lịch sử trống ở thread đó.

Kết quả chứng minh persistent memory hữu ích trên hai dataset đã cung cấp.
100% không chứng minh agent hiểu mọi cách diễn đạt tiếng Việt: responder và
extractor offline chỉ hỗ trợ một số mẫu, còn scorer không hiểu phủ định.

## 2. Vì sao Advanced có thể tốn hơn ở hội thoại ngắn?

Ở Standard, Advanced xử lý 28182 prompt token, cao hơn Baseline 18433 là 9749
token, tương đương **52,89%**. Không có lần compact nào; chi phí chèn profile
và chỉ dẫn sử dụng profile xuất hiện ở mỗi lượt nhưng chưa được bù bằng nén
lịch sử. Chi phí này đổi lấy khả năng recall qua phiên mới.

Output của Advanced lại thấp hơn: 1523 so với 1834 token, giảm 311 token.
Trong mô phỏng này, câu trả lời phụ thuộc vào fact đã có và cách định dạng;
không thể suy từ output token rằng profile bị bỏ qua hay compact tiết kiệm
token sinh ra. Ghi file không tự tạo ra model output token. Lượng profile được
đưa vào ngữ cảnh được tính ở `Prompt tokens processed`.

## 3. Vì sao compact có lợi ở hội thoại dài?

Prompt từng lượt chat của Baseline trong Stress tăng từ 222 lên 3010 token;
tổng gồm recall là 27257. Lịch sử dài tiếp tục được mang theo qua các lượt.
Advanced compact 4 lần, giữ summary giới hạn và message gần nhất, còn 12617
prompt token: giảm **53,71%** so với Baseline.

| Biến thể Stress | Output token | Prompt token | Recall | Profile tăng (byte) | Compactions |
|---|---:|---:|---:|---:|---:|
| Advanced bật compact | 822 | 12617 | 100% | 238 | 4 |
| Advanced tắt compact | 822 | 28618 | 100% | 238 | 0 |

Khi chỉ nâng ngưỡng lên 1000000, Advanced tốn 28618 prompt token, còn cao hơn
Baseline vì vẫn mang profile. Bật compact giảm **55,91%** so với chính Advanced
tắt compact; output, recall, quality và profile cuối không đổi. Phép đối chứng
này xác định lợi ích chủ yếu nằm ở ngữ cảnh đầu vào. Ngưỡng thử chỉ thuộc bản
sao config; cấu hình chạy thông thường vẫn là 1000.

Giới hạn: summary heuristic có thể mất chi tiết, và compact không nén chính
profile. Ngưỡng là điều kiện kích hoạt, không phải giới hạn cứng cho toàn bộ
prompt; system prompt, profile và message mới quá dài vẫn tạo chi phí riêng.

## 4. Memory file tăng trưởng ra sao và có rủi ro gì?

Profile cuối Standard tăng 316 byte, Stress tăng 238 byte; Baseline bằng 0.
Đây là chênh lệch kích thước UTF-8 cuối trừ đầu, đếm mỗi user một lần trong mỗi
bộ, không phải tổng số byte đã ghi hay tốc độ tăng theo thời gian. Hai bộ dùng
hai user khác nhau nên không thể suy thành đường cong tăng trưởng dài hạn.

`upsert_facts()` thay dòng theo key và loại bản trùng của key đang cập nhật,
tránh nối thêm mọi correction. Điều này hạn chế tăng trưởng nhưng chưa có
trần kích thước profile, TTL hay memory decay. Nhiều field, giá trị dài hoặc
ghi chú Markdown vẫn có thể khiến file và prompt tăng. Bản hiện tại cũng chưa
có khóa giao dịch cho nhiều process ghi cùng profile hay lịch sử rollback.

Regex bỏ qua câu hỏi, chuyến họp ngắn và câu đùa trong dataset. Ở mục 10, probe
`Mình ở Đà Nẵng, trước đây mình ở Huế.` và `Bạn tôi tên là An.` đã làm lộ lỗi
nhầm quá khứ/chủ thể. Mục 11 bổ sung guard cho mệnh đề lịch sử và chủ thể sở hữu
ngôi thứ ba, cùng test hồi quy: hiện tại hai probe lần lượt trả nơi ở Đà Nẵng
và không có update. JSON bonus đã được đo lại với `matches_expected=true`.

Đây là sửa hai nhóm mẫu cụ thể, không phải bộ phân tích chủ thể/thời gian tổng
quát. Trích sai ở cách diễn đạt khác vẫn có thể làm fact sai tồn tại qua phiên
và được ưu tiên hơn summary. Cần nguồn fact và xác nhận khi mơ hồ trước khi dùng
ngoài phạm vi lab. Profile chứa dữ liệu cá nhân nên còn cần quyền truy cập và
cơ chế xóa phù hợp.

## Bonus đã chọn: Conflict handling

Correction có trong input thật: Standard đổi Đà Nẵng thành Huế và backend
engineer thành MLOps engineer; Stress đổi Huế thành Đà Nẵng. Lưu bền vững mà
không sửa fact cũ sẽ khiến agent nhớ sai thông tin hiện tại.

Cơ chế đã triển khai ở các mục trước: extractor lấy assertion mới, store thay
giá trị theo key và tránh dòng trùng, profile hiện tại được ưu tiên hơn summary
và lịch sử cũ. Cập nhật style từng phần giữ lại những ràng buộc không bị đổi.
Mục 10 chọn cơ chế này làm bonus và bổ sung phép đo tái lập được; không thêm
confidence score giả hay tuyên bố đã có memory decay.

`benchmark_conflicts.py` so sánh hai Advanced trên nguyên dataset: biến thể
đối chứng giữ giá trị đầu tiên riêng cho `location`/`profession`, biến thể còn
lại cập nhật correction như code hiện tại. Extractor, compact, responder,
các field khác và cách chấm giữ nguyên; mỗi biến thể dùng profile tạm sạch.

| Bộ | Chính sách | Recall | Quality | Prompt token | Profile tăng (byte) |
|---|---|---:|---:|---:|---:|
| Standard | Giữ giá trị đầu tiên | 64,3% | 0,668 | 28297 | 324 |
| Standard | Conflict handling | 100% | 1,000 | 28182 | 316 |
| Stress | Giữ giá trị đầu tiên | 66,7% | 0,750 | 12603 | 232 |
| Stress | Conflict handling | 100% | 1,000 | 12617 | 238 |

Bonus giải quyết fact hiện tại bị khóa ở giá trị cũ: recall tăng khoảng **35,7
điểm phần trăm** ở Standard và **33,3 điểm phần trăm** ở Stress. Đây là điểm
recall ba mức trung bình, không phải tỷ lệ câu trả lời hoàn toàn đúng. Lợi ích
chính là recall; prompt giảm nhẹ ở Standard nhưng tăng 14 token ở Stress, nên
không có bằng chứng bonus luôn giảm token cost.

Rủi ro thêm vào: assertion mới trích sai có thể ghi đè fact đúng; “xuất hiện
sau” chưa đồng nghĩa “đúng hiện tại”. Guard quá khứ/chủ thể ở mục 11 xử lý các
mẫu đã kiểm tra; profile ưu tiên mạnh vẫn có thể làm lỗi chưa nhận diện lan qua
nhiều phiên. Regex cũng có thể bỏ qua correction ngoài mẫu hỗ trợ. Chính sách giữ giá trị đầu tiên là đối
chứng có chủ đích, không đại diện cho mọi hệ thống thiếu conflict handling.

## Kiểm chứng và đối chiếu rubric

Các test hiện có kiểm tra correction loại dòng trùng, profile mới thắng summary
cũ, câu hỏi/nhiễu không sửa profile và merge style. Test mới cho phép đo bonus
kiểm tra chỉ hai field mục tiêu khác nhau, cập nhật tên vẫn hoạt động ở cả hai
biến thể, kết quả tái lập, không gọi live model và không sửa profile/input thật.

| Mốc rubric | Bằng chứng trong repo |
|---|---|
| 0–60 | Hai agent, persistent profile, compact và dataset |
| 60–75 | Input công bằng, bảng đủ sáu cột, bốn test lab |
| 75–90 | Hai benchmark, đo tắt compact, bốn phần phân tích ở trên |
| 90–100 | Conflict handling, số đo recall và rủi ro cụ thể |

Đây là đối chiếu bằng chứng, không phải cam kết điểm chấm. Live API và chất
lượng trả lời tổng quát chưa được benchmark; kết quả thuộc mô phỏng offline.

Chạy từ thư mục gốc:

```bash
source .venv/bin/activate
python src/benchmark.py
python src/benchmark_conflicts.py --output results/conflict_results.json
python -m pytest -q
```

Output benchmark, kết quả bonus và file phân tích được lưu trong `results/`.
`state/` là dữ liệu sinh ra và đã được gitignore. Phần chuẩn bị mục 11 gồm
dependencies được ghim, 138 test pass và CI chạy offline; người học tự nộp
link repo trên VLearn.
