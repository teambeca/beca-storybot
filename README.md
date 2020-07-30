# BECA Takımı

BECA ekibi olarak amacımız, NLP projeleri için gerekli veri setlerini hem interaktif bir şekilde kullanıcılardan toplayan, hem de daha sonrasında doğal dil işleme ile bu verileri eşsiz hikayeler yazmak için kullanan, Baazi uygulamasını ülkemize kazandırmaktır.

# Baazi

Doğal dil işleme bir yapay zeka alt dalıdır ve içinde bir çok dil işleme tekniği bulundurur. Baazi projesindeki asıl amaç, doğal dil işleme için proje geliştiricilerine zenginleştirilmiş veri seti olanağı sağlamaktır.

BECA takımı ve Baazi uygulaması hakkında daha bilgi almak için, [tıklayın](https://www.google.com/).

## Başlangıç

BECA StoryBot, Python dilinde yazılmıştır.

```trainer.py``` yapay zeka modelini oluşturmakta ve eğitmektedir. Klasik bir RNN yapısı kullanılmaktadır.

Başlangıç parametreleri ve açıklamaları aşağıdaki gibidir.

Parametre | Açıklama
----------|------------
sq_length|dizi uzunluğu
step_size|adım boyutu
input_folder|dataset dosyası
output_folder|modelin kayıt edileceği dosya
batch_size|parti boyutu
epochs|dönem

## Gereksinimler

StoryBot'u eğitebilmek için için, sisteminize Python'un (version : 3.8) kurulu olması gerkemektedir.

Gerekli tüm kütüphaneleri kurmak için Anaconda veya Miniconda kullanarak, aşağıda yazılı komutu verebilirsiniz.

```
conda env create --file beca.yml
```

Eğitmiş olduğunuz modelinizi test edebilmek için ```predicter.py``` scriptini kullanabilirsiniz.

## Lisans
[Apache License 2.0](LICENSE)
