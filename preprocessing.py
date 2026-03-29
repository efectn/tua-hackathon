import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import StandardScaler

# 1. VERİYİ YÜKLEME
def load_and_create_df(file_path):
    with open(file_path, 'r') as f:
        json_data = json.load(f)
        
    # Sütun isimlerini parameters listesinden çekiyoruz
    columns = [param['name'] for param in json_data['parameters']]
    
    # Veriyi DataFrame'e dönüştürüyoruz
    df = pd.DataFrame(json_data['data'], columns=columns)
    
    return df

# 2. VERİ TEMİZLİĞİ (Data Cleaning)
def clean_data(df):
    df_clean = df.copy()
    
    # 'Time' sütununu datetime formatına çevirip index yapıyoruz
    df_clean['Time'] = pd.to_datetime(df_clean['Time'])
    df_clean.set_index('Time', inplace=True)
    
    # NASA OMNI verisindeki "Kayıp Veri" (Fill) değerlerini sözlük olarak tanımlıyoruz
    fill_values = {
        'F': 9999.99,
        'BZ_GSM': 9999.99,
        'flow_speed': 99999.9,
        'proton_density': 999.99,
        'T': 9999999.0,
        'E': 999.99,
        'SYM_H': 99999
    }
    
    # Fill değerlerini NaN (Not a Number) ile değiştiriyoruz
    for col, val in fill_values.items():
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].replace(val, np.nan)
            
    # Zaman serisi olduğu için eksik değerleri Lineer İnterpolasyon ile dolduruyoruz
    # İnterpolasyonun yetişemediği ilk/son satırlar için ileri/geri doldurma (ffill/bfill) yapıyoruz
    df_clean = df_clean.interpolate(method='time').ffill().bfill()
    
    return df_clean

# 3. ÖZELLİK MÜHENDİSLİĞİ (Feature Engineering)
def engineer_features(df):
    df_feat = df.copy()
    
    # a. Sadece Güneye Yönelik Manyetik Alan (Southward IMF)
    # Fırtınaları tetikleyen şey BZ'nin negatif olmasıdır. Pozitifleri 0 yapıp sadece negatif etkiyi alıyoruz.
    df_feat['Bz_south'] = df_feat['BZ_GSM'].apply(lambda x: x if x < 0 else 0)
    
    # b. Enerji Transfer Fonksiyonu (V * Bz_south)
    # Güneş rüzgarı hızı ile güneye yönelik manyetik alanın çarpımı, Dünya'ya aktarılan enerjinin iyi bir temsilidir.
    df_feat['V_Bz_south'] = df_feat['flow_speed'] * df_feat['Bz_south']
    
    # c. Hareketli Ortalamalar (Rolling Averages)
    # Bir fırtınanın oluşması için manyetik alanın *uzun süre* güneye bakması gerekir.
    # Bu yüzden 30 dakikalık (30 satır) ve 60 dakikalık hareketli ortalamalar ekliyoruz.
    windows = [30, 60]
    for w in windows:
        df_feat[f'BZ_GSM_roll_{w}'] = df_feat['BZ_GSM'].rolling(window=w).mean()
        df_feat[f'flow_speed_roll_{w}'] = df_feat['flow_speed'].rolling(window=w).mean()
        df_feat[f'proton_density_roll_{w}'] = df_feat['proton_density'].rolling(window=w).mean()
        
    # Hareketli ortalamalardan kaynaklı ilk satırlardaki NaN değerlerini dolduruyoruz
    df_feat = df_feat.bfill()
    
    return df_feat

# 4. ÖLÇEKLENDİRME (Scaling)
def scale_features(df):
    df_scaled = df.copy()
    
    # Hedef değişkenimiz SYM_H hariç tüm özellikleri standartlaştırıyoruz (Z-Score Normalization)
    features_to_scale = [col for col in df_scaled.columns if col != 'SYM_H']
    
    scaler = StandardScaler()
    df_scaled[features_to_scale] = scaler.fit_transform(df_scaled[features_to_scale])
    
    return df_scaled, scaler

# KODU ÇALIŞTIRMA AKIŞI
if __name__ == "__main__":
    # 1. Yükle
    df_raw = load_and_create_df('data_test.json')
    
    # 2. Temizle
    df_cleaned = clean_data(df_raw)
    
    # 3. Özellik Üret
    df_engineered = engineer_features(df_cleaned)
    
    # 4. Ölçeklendir (Modele vermeye hazır son hal)
    df_final, fitted_scaler = scale_features(df_engineered)
    
    print("Veri ön işleme tamamlandı!")
    print(f"Veri Seti Boyutu: {df_final.shape}")
    print(df_final.head())
    
    df_final.to_csv("preprocessed_data_test.csv", index=True)