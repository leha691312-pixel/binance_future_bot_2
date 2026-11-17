MERGED BOT PACKAGE — 2026-11-17 09:33 UTC
================================================================
Gộp TẤT CẢ fix từ top-N volatility đến hiện tại (không đè mất nhau).

INCLUDED
--------
1. core/exchange.py
   - Top N theo BIẾN ĐỘNG 24h (|%| + range), lọc min quote volume
   - close_position reduceOnly (sửa -4164 khi timeout đóng lệnh nhỏ)
   - TRADING symbol filter

2. agents/execution.py
   - Dynamic risk (DD / streak / vol / conf / exposure)
   - Min notional 5 USDT (-4164) bump/skip
   - Clamp max qty (-4005)
   - Protective orders flow

3. agents/risk_gate.py
   - Max-hold snapshot (giữ open_time/style khi sync)
   - RR chỉ lọc (structure-aware)
   - TF→style: 15m scalp | 1h/2h/4h day | 6h+ swing

4. agents/signal_refine.py
   - Consensus + HTF/MTF bias + Killzone (UTC)

5. agents/structure_tp.py + atr_utils.py
6. agents/strategies/* — structure TP, RR thấp hơn, ATR tối ưu, EMARSIMACDDay
7. agents/ml_agent.py — per-style nếu có trong zip nguồn
8. main.py — restart Windows-safe (nếu compile OK)
9. config/risk.yaml — dynamic_risk + atr_profiles + min RR 1.2

INSTALL (Windows)
-----------------
1. Backup toàn bộ thư mục bot hiện tại
2. Copy đè các file trong zip (giữ nguyên .env của bạn)
3. Merge thủ công settings.yaml:
   - Giữ enabled_strategies / API mode của bạn
   - Thêm min_quote_volume_top: 500000
   - Copy block signal_refine từ config/signal_refine_fragment.yaml
   - Tham khảo settings_MERGED_REFERENCE.yaml cho strategy_params RR/ATR
4. python -c "import yaml; yaml.safe_load(open('config/settings.yaml',encoding='utf-8')); print('YAML OK')"
5. python main.py

VERIFY LOG
----------
- "Top symbols by 24h volatility"  → đúng top N biến động
- "Dynamic risk ..."               → dynamic risk còn
- "TIMEOUT ... close" không -4164  → reduceOnly close
- Killzone/MTF log nếu bật

KHÔNG copy đè .env / API keys.
