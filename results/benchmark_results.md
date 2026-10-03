# Mục 9 — Kết quả chạy benchmark và kiểm chứng compact

Chạy offline bằng Python 3.12.14; ngưỡng compact thực tế 1000 token,
giữ 4 message mới nhất. Token là ước lượng ký tự, gồm cả lượt chat và recall.
Quality là tỷ lệ chuỗi fact kỳ vọng xuất hiện (0..1), không phải đánh giá của judge model.

## Hai lần chạy sạch

Đã chạy `python src/benchmark.py` hai lần trong hai process độc lập, với
`PYTHONHASHSEED=1` và `2`. Output giống nhau hoàn toàn; profile cuối mỗi lần cũng giống nhau.
`main()` xóa profile của user dataset trong namespace benchmark trước mỗi lần chạy,
và tạo agent mới; profile ngoài namespace benchmark không bị thay đổi.
Không cần xóa toàn bộ `state/`.
SHA-256 output: `0ad2ee95072889b0ff4cd57310af5df6d41ade19845270dacc6bf48835df38e4`.

Offline benchmark: estimated tokens; quality = expected-fact coverage (0..1).
Token totals include chat and recall. Each conversation uses a fresh recall thread.

## Standard Benchmark

| Agent    | Agent tokens only   | Prompt tokens processed   | Cross-session recall   | Response quality   | Memory growth (bytes)   | Compactions   |
|----------|---------------------|---------------------------|------------------------|--------------------|-------------------------|---------------|
| Baseline | 1834                | 18433                     | 0.0%                   | 0.000              | 0                       | 0             |
| Advanced | 1523                | 28182                     | 100.0%                 | 1.000              | 316                     | 0             |

## Long-Context Stress Benchmark

| Agent    | Agent tokens only   | Prompt tokens processed   | Cross-session recall   | Response quality   | Memory growth (bytes)   | Compactions   |
|----------|---------------------|---------------------------|------------------------|--------------------|-------------------------|---------------|
| Baseline | 826                 | 27257                     | 0.0%                   | 0.000              | 0                       | 0             |
| Advanced | 822                 | 12617                     | 100.0%                 | 1.000              | 238                     | 4             |

## Tắt compact trên cùng bộ stress

Dùng agent mới và thư mục tạm riêng cho từng biến thể. Chỉ thay ngưỡng compact
thành 1.000.000 trong bản sao config của Advanced, giữ nguyên input, cách đặt thread,
thứ tự chat/recall, công thức chấm và estimator.

| Agent                       | Agent tokens only   | Prompt tokens processed   | Cross-session recall   | Response quality   | Memory growth (bytes)   | Compactions   |
|-----------------------------|---------------------|---------------------------|------------------------|--------------------|-------------------------|---------------|
| Baseline                    | 826                 | 27257                     | 0.0%                   | 0.000              | 0                       | 0             |
| Advanced                    | 822                 | 12617                     | 100.0%                 | 1.000              | 238                     | 4             |
| Advanced (compact disabled) | 822                 | 28618                     | 100.0%                 | 1.000              | 238                     | 0             |

## Đối chiếu số liệu

- Baseline recall 0% và memory growth 0 byte ở cả hai bộ: không nhớ qua thread mới.
- Advanced recall 100%, profile tăng 316 byte ở Standard và 238 byte ở Stress.
- Trong 16 lượt chat của Stress, prompt mỗi lượt của Baseline tăng từ
  222 lên 3010 token.
  Tổng prompt gồm cả recall là 27257.
- Advanced compact 4 lần, còn 12617 prompt token:
  giảm 53.71% so với Baseline. Tắt compact làm prompt tăng lên
  28618, vượt Baseline do còn mang profile vào ngữ cảnh.
  So với Advanced tắt compact, bật compact giảm 55.91% prompt token.
- Khi bật/tắt compact, output token của Advanced cùng bằng 822,
  recall cùng 100%, quality cùng 1 và profile cùng tăng 238 byte.
  Chênh lệch trong phép thử này nằm ở prompt context.
- Standard không compact; prompt Advanced 28.182 cao hơn Baseline 18.433 vì chi phí
  profile chưa được bù lại bởi việc nén lịch sử.

Hai dataset và config được kiểm tra giữ nguyên. Ngưỡng thử nghiệm chỉ tồn tại trong
bản sao config của process kiểm chứng; ngưỡng chạy thông thường vẫn là
1000. Không chỉnh dữ liệu để thay kết quả.

Chi tiết từng lượt, SHA-256 dataset/profile và số liệu thô: [benchmark_results.json](benchmark_results.json).

## Chạy lại

```bash
source .venv/bin/activate
python src/benchmark.py
```

Phép thử tắt compact có thể chạy bằng override chỉ cho một lệnh:

```bash
COMPACT_THRESHOLD_TOKENS=1000000 python src/benchmark.py
python src/benchmark.py
```

Lệnh thứ hai chạy lại với cấu hình thông thường và tái tạo profile benchmark.
File này ghi số liệu kiểm chứng mục 9. Phân tích đầy đủ và bonus ở mục 10 đã được
lưu tại [memory_analysis.md](memory_analysis.md).
