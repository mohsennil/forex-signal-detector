import requests
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import json
import os
import sys

class GitHubForexDetector:
    def __init__(self):
        # تنظیمات
        self.fast_ma = 20
        self.slow_ma = 50
        self.max_pullback = 3
        self.min_rsi_change = 10.0
        self.rsi_period = 14
        
        self.forex_pairs = ['EURUSD', 'GBPUSD', 'AUDUSD', 'USDCHF', 'USDCAD']
        self.historical_data = {}
        self.last_signals = {}
        self.signal_count = 0
        
        # برای GitHub Actions
        self.is_github_actions = os.getenv('GITHUB_ACTIONS') is not None
        self.artifacts_dir = os.getenv('GITHUB_WORKSPACE', '.')
        self.signals_file = os.path.join(self.artifacts_dir, 'signals.json')
        self.summary_file = os.path.join(self.artifacts_dir, 'summary.md')
        
        print("🚀 سیستم تشخیص پولبک - GitHub Actions")
        print("=" * 60)
        
        self.initialize_historical_data()

    def initialize_historical_data(self):
        print("📡 دریافت قیمت‌های فعلی...")
        for pair in self.forex_pairs:
            current_price_data = self.get_yahoo_live_price(pair)
            if current_price_data:
                base_price = current_price_data['price']
                print(f"   ✅ {pair}: {base_price:.5f}")
            else:
                default_prices = {
                    'EURUSD': 1.16120, 'GBPUSD': 1.31810, 'AUDUSD': 0.65660,
                    'USDCHF': 0.80010, 'USDCAD': 1.39620
                }
                base_price = default_prices[pair]
                print(f"   ⚠️ {pair}: استفاده از قیمت پیش‌فرض {base_price:.5f}")
            
            self.historical_data[pair] = self.generate_historical_data(base_price, pair)
            self.last_signals[pair] = None

    def get_yahoo_live_price(self, pair):
        try:
            symbol = f"{pair}=X"
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'application/json'
            }
            
            response = requests.get(url, headers=headers, timeout=15)
            data = response.json()
            
            if 'chart' in data and 'result' in data['chart']:
                result = data['chart']['result'][0]
                current_price = result['meta']['regularMarketPrice']
                return {'price': current_price, 'timestamp': datetime.now()}
        except Exception as e:
            print(f"   ❌ خطا در دریافت قیمت {pair}: {e}")
        return None

    def calculate_ema(self, data, period):
        return data.ewm(span=period, adjust=False).mean()

    def calculate_rsi(self, data, period=14):
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)
        
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def check_pullback_conditions(self, df, current_index):
        try:
            if current_index < self.max_pullback + 5:
                return "NO_DATA"
                
            current = df.iloc[current_index]
            is_downtrend = current['fast_ema'] < current['slow_ema']
            is_uptrend = current['fast_ema'] > current['slow_ema']
            
            if not (is_downtrend or is_uptrend):
                return "NO_TREND"
            
            pullback_candles = []
            for j in range(1, self.max_pullback + 1):
                if current_index - j < 0:
                    return "NO_DATA"
                prev_candle = df.iloc[current_index - j]
                pullback_candles.append(prev_candle)
            
            if is_downtrend:
                is_pullback = all(candle['close'] > candle['open'] for candle in pullback_candles)
                if (is_pullback and current['close'] > current['fast_ema'] and 
                    current['rsi'] > (50 + self.min_rsi_change) and current['close'] > current['open']):
                    return "SELL"
            
            elif is_uptrend:
                is_pullback = all(candle['close'] < candle['open'] for candle in pullback_candles)
                if (is_pullback and current['close'] < current['fast_ema'] and 
                    current['rsi'] < (50 - self.min_rsi_change) and current['close'] < current['open']):
                    return "BUY"
            
            return "NO_SIGNAL"
        except Exception as e:
            print(f"خطا در بررسی شرایط: {e}")
            return "ERROR"

    def generate_historical_data(self, base_price, pair):
        dates = pd.date_range(start=datetime.now() - timedelta(days=30), periods=200, freq='15min')
        np.random.seed(int(datetime.now().timestamp()) % 1000 + hash(pair) % 1000)
        
        prices = [base_price]
        for i in range(1, 200):
            change = np.random.normal(0, base_price * 0.0003)
            prices.append(prices[-1] + change)
        
        df_data = []
        for i, date in enumerate(dates):
            if i >= len(prices): break
            base_px = prices[i]
            volatility = base_px * 0.0001
            
            open_price = base_px
            close_price = prices[i] if i == len(prices)-1 else base_px + np.random.normal(0, volatility)
            
            high_price = max(open_price, close_price) + abs(np.random.exponential(volatility))
            low_price = min(open_price, close_price) - abs(np.random.exponential(volatility))
            
            df_data.append({
                'datetime': date, 'open': open_price, 'high': high_price,
                'low': low_price, 'close': close_price, 'volume': np.random.randint(1000,5000)
            })
        
        return pd.DataFrame(df_data).sort_values('datetime').reset_index(drop=True)

    def update_data_with_live_price(self, pair, live_price):
        df = self.historical_data[pair]
        last_close = df['close'].iloc[-1]
        
        new_row = df.iloc[-1].copy()
        new_row['datetime'] = datetime.now()
        new_row['open'] = last_close
        new_row['close'] = live_price
        
        if live_price > last_close:
            new_row['high'] = live_price * 1.0001
            new_row['low'] = last_close * 0.9998
        else:
            new_row['high'] = last_close * 1.0001
            new_row['low'] = live_price * 0.9998
        
        new_row['volume'] = np.random.randint(1000, 5000)
        self.historical_data[pair] = pd.concat([df.iloc[1:], pd.DataFrame([new_row])], ignore_index=True)
        
        return live_price - last_close

    def save_signal(self, pair, signal, price, rsi, trend):
        """ذخیره سیگنال در فایل"""
        signal_data = {
            'pair': pair,
            'signal': signal,
            'price': price,
            'rsi': rsi,
            'trend': trend,
            'timestamp': datetime.now().isoformat(),
            'count': self.signal_count
        }
        
        try:
            with open(self.signals_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(signal_data, ensure_ascii=False) + '\n')
            print(f"   💾 سیگنال {pair} {signal} ذخیره شد")
        except Exception as e:
            print(f"   ❌ خطا در ذخیره: {e}")

    def create_summary(self, all_signals):
        """ایجاد فایل خلاصه"""
        try:
            summary_content = [
                "# 📊 خلاصه سیگنال‌های فارکس",
                f"**تاریخ تولید:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                f"**تعداد سیگنال‌ها:** {len(all_signals)}",
                "",
                "## 🎯 سیگنال‌های شناسایی شده:",
                ""
            ]
            
            for pair, signal, current, trend, price in all_signals:
                summary_content.extend([
                    f"### {pair} - {signal}",
                    f"- **قیمت:** {price:.5f}",
                    f"- **RSI:** {current['rsi']:.1f}",
                    f"- **روند:** {trend}",
                    f"- **زمان:** {datetime.now().strftime('%H:%M:%S')}",
                    ""
                ])
            
            if not all_signals:
                summary_content.append("⚠️ هیچ سیگنالی در این اجرا شناسایی نشد")
            
            with open(self.summary_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(summary_content))
                
            print("   📄 فایل خلاصه ایجاد شد")
        except Exception as e:
            print(f"   ❌ خطا در ایجاد خلاصه: {e}")

    def analyze_pair(self, pair):
        print(f"🔎 تحلیل {pair}:")
        
        live_data = self.get_yahoo_live_price(pair)
        if not live_data:
            return "NO_DATA", None, None, None
        
        live_price = live_data['price']
        price_change = self.update_data_with_live_price(pair, live_price)
        
        df = self.historical_data[pair]
        df['fast_ema'] = self.calculate_ema(df['close'], self.fast_ma)
        df['slow_ema'] = self.calculate_ema(df['close'], self.slow_ma)
        df['rsi'] = self.calculate_rsi(df['close'], self.rsi_period)
        
        current = df.iloc[-1]
        
        change_icon = "🟢" if price_change >= 0 else "🔴"
        print(f"   💰 قیمت: {live_price:.5f} {change_icon}")
        print(f"   📊 RSI: {current['rsi']:.1f}")
        
        if current['fast_ema'] < current['slow_ema']:
            trend = "نزولی 📉"
        else:
            trend = "صعودی 📈"
        
        signal = self.check_pullback_conditions(df, -1)
        
        if signal in ["BUY", "SELL"]:
            if self.last_signals[pair] == signal:
                return "DUPLICATE", current, trend, live_price
            self.last_signals[pair] = signal
        
        return signal, current, trend, live_price

    def run_single_check(self):
        """اجرای یک چک کامل"""
        print(f"\n🔄 اجرا در: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("-" * 50)
        
        all_signals = []
        
        for pair in self.forex_pairs:
            signal, current, trend, price = self.analyze_pair(pair)
            
            if signal in ["BUY", "SELL"]:
                all_signals.append((pair, signal, current, trend, price))
                print(f"   🎯 {pair} {signal} شناسایی شد!")
            else:
                if signal != "DUPLICATE":
                    print(f"   📊 {pair}: {signal}")
        
        if all_signals:
            print(f"\n🎯 {len(all_signals)} سیگنال شناسایی شد")
            
            for pair, signal, current, trend, price in all_signals:
                self.signal_count += 1
                self.save_signal(pair, signal, price, current['rsi'], trend)
            
            # ایجاد خلاصه
            self.create_summary(all_signals)
        else:
            print(f"\n📊 هیچ سیگنالی در این چک شناسایی نشد")
            
            # ایجاد خلاصه خالی
            self.create_summary([])
        
        return len(all_signals)

if __name__ == "__main__":
    detector = GitHubForexDetector()
    signals_found = detector.run_single_check()
    
    # خروج با کد مناسب برای GitHub Actions
    sys.exit(0 if signals_found >= 0 else 1)
