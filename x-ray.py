# -*- coding: utf-8 -*-
"""
YOLOv8 ile X-ray Bagaj Anomali Tespiti Eğitimi
@author: emrek (ve Gemini)
"""

from ultralytics import YOLO
import torch
import os
import yaml # YAML dosyasını kontrol etmek için (isteğe bağlı ama faydalı)
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

def tren_yolo_modeli():
    """
    YOLOv8 modelini X-ray bagaj veri kümesiyle eğitir.
    """
    print("YOLOv8 X-ray Anomali Tespiti Eğitimi Başlatılıyor...")

    # 1. Cihazı Belirle (GPU varsa GPU, yoksa CPU)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Kullanılacak Cihaz: {device}")
    if device == 'cpu':
        print("UYARI: CPU üzerinde eğitim çok daha yavaş olacaktır. Mümkünse GPU kullanın.")

    # 2. data.yaml Dosyasının Yolu
    # Bu YAML dosyasının, Python script'inizin çalıştığı dizine göre
    # doğru bir yolda olduğundan emin olun.
    # Eğer script ile aynı dizindeyse: data_yaml_path = 'xray_data.yaml' (veya verdiğiniz isim)
    # Önceki mesajınızdaki içeriğe göre dosya adını 'xray_data.yaml' varsayıyorum.
    data_yaml_path = 'data.yaml' # LÜTFEN BU YOLU KONTROL EDİN VE GEREKİRSE GÜNCELLEYİN!

    # YAML dosyasının varlığını ve temel içeriğini kontrol edelim
    if not os.path.exists(data_yaml_path):
        print(f"HATA: '{data_yaml_path}' bulunamadı!")
        print("Lütfen YAML dosyasının doğru yolda olduğundan ve adının doğru yazıldığından emin olun.")
        return
    
    try:
        with open(data_yaml_path, 'r', encoding='utf-8') as f: # encoding eklendi
            data_config = yaml.safe_load(f)
            print("\n--- Yüklenen data.yaml İçeriği ---")
            print(f"Eğitim Görüntüleri Yolu: {data_config.get('train')}")
            print(f"Doğrulama Görüntüleri Yolu: {data_config.get('val')}")
            print(f"Test Görüntüleri Yolu (varsa): {data_config.get('test')}")
            print(f"Sınıf Sayısı (nc): {data_config.get('nc')}")
            print(f"Sınıf İsimleri (names): {data_config.get('names')}")
            print("---------------------------------\n")
            if not data_config.get('train') or not data_config.get('val'):
                print("HATA: YAML dosyasında 'train' veya 'val' yolları eksik!")
                return
            if data_config.get('nc') != len(data_config.get('names')):
                print("UYARI: YAML'deki 'nc' ile 'names' listesinin uzunluğu tutarsız!")
            if any(name.isdigit() for name in data_config.get('names')):
                print("UYARI: YAML dosyasındaki sınıf isimleri ('names') rakamlardan oluşuyor.")
                print("Anlamlı sonuçlar için bu isimleri 'gun', 'knife' gibi gerçek nesne isimleriyle güncellemeniz önerilir.")

    except Exception as e:
        print(f"HATA: '{data_yaml_path}' dosyası okunurken bir sorun oluştu: {e}")
        return

    # 3. Kullanılacak YOLO Modelini Seç
    # Farklı boyutlarda ve hızlarda modeller mevcuttur:
    # yolov8n.pt (nano - en hızlı, en küçük)
    # yolov8s.pt (small - iyi bir denge)
    # yolov8m.pt (medium)
    # yolov8l.pt (large)
    # yolov8x.pt (extra large - en yavaş, en yüksek başarım potansiyeli)
    model_secimi = 'yolov8s.pt' # Başlangıç için 's' veya 'n' iyi bir seçenektir.
    print(f"'{model_secimi}' modeli kullanılacak.")

    # 4. Eğitim Parametreleri
    EPOCHS = 100        # Toplam eğitim turu (deneyerek artırılabilir, örn: 100, 200)
    BATCH_SIZE = 32    # Tek seferde işlenecek görüntü sayısı (GPU belleğinize göre ayarlayın, örn: 8, 16, 32)
                       # Eğer bellek hatası alırsanız bu değeri düşürün.
    IMAGE_SIZE = 416   # Görüntülerin yeniden boyutlandırılacağı boyut (YAML'deki verilerle de ilişkili olabilir)
                       # Modelin varsayılanı genellikle 640'tır, ancak 416 da yaygındır.
    PROJE_ADI = 'xray_nesne_tespiti2' # Sonuçların kaydedileceği proje klasörü
    DENEY_ADI = 'yolov8s_ilk_deney2' # Bu özel eğitim için alt klasör adı

    # 5. YOLO Modelini Yükle
    # '.pt' dosyası hem modelin mimarisini hem de önceden eğitilmiş ağırlıklarını içerir.
    # Bu, transfer öğrenme yapmamızı sağlar.
    try:
        model = YOLO(model_secimi)
        print(f"'{model_secimi}' modeli başarıyla yüklendi.")
    except Exception as e:
        print(f"HATA: YOLO modeli yüklenirken bir sorun oluştu: {e}")
        print("Ultralytics kütüphanesinin doğru kurulduğundan ve model adının doğru olduğundan emin olun.")
        return

    # 6. Modeli Eğit
    print("\nModel eğitimi başlatılıyor...")
    try:
        results = model.train(
            data=data_yaml_path,    # data.yaml dosyasının yolu
            epochs=EPOCHS,
            batch=BATCH_SIZE,
            imgsz=IMAGE_SIZE,
            device=device,          # 'cuda' veya 'cpu'
            patience=25,            # Erken durdurma için sabır (performans artmazsa 10 epoch sonra durur)
            project=PROJE_ADI,      # Kayıt dizini: runs/detect/PROJE_ADI
            name=DENEY_ADI,         # Kayıt dizini: runs/detect/PROJE_ADI/DENEY_ADI
            exist_ok=False          # Eğer aynı isimde bir deney varsa üzerine yazma, hata ver (True yaparsanız üzerine yazar)
        )
        print("\nEğitim tamamlandı!")
        # En iyi modelin yolu genellikle results.save_dir + '/weights/best.pt' şeklinde olur.
        best_model_path = os.path.join(results.save_dir, 'weights', 'best.pt')
        print(f"✅ En iyi model ağırlıkları: {best_model_path}")

        # Eğitim metriklerini grafik olarak göster (results.png)
        results_img_path = os.path.join(results.save_dir, 'results.png')
        if os.path.exists(results_img_path):
            print("📊 Eğitim metrikleri görseli gösteriliyor...")
            img = Image.open(results_img_path)
            plt.imshow(img)
            plt.axis('off')
            plt.title("Eğitim Metrikleri")
            plt.show()
        else:
            print("results.png bulunamadı!")

        # Eğitim istatistiklerini (CSV) göster
        csv_path = os.path.join(results.save_dir, 'results.csv')
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            print("\n📈 Eğitim Sonuçları (Son Epoch):")
            print(df.tail(1)[['epoch', 'metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95']])
        else:
            print("results.csv bulunamadı!")

    except Exception as e:
        print(f"Eğitim sırasında bir hata oluştu: {e}")
        print("Lütfen YAML dosyasındaki yolların ve veri formatının doğru olduğundan emin olun.")
        print("Ayrıca GPU belleğini ve CUDA/PyTorch kurulumunu kontrol edin.")

if __name__ == '__main__':
    # Bu satır, script doğrudan çalıştırıldığında ana fonksiyonu çağırır.
    tren_yolo_modeli()
