# FaceTrackEd

Gudra seju atpazīšanas sistēma skolēnu uzskaitei un statistikai. Projekts veidots ar Python, izmantojot `face_recognition`, `OpenCV` un `Pandas`. Lietotne automātiski atpazīst skolēnus no attēliem vai video un apkopo datus par viņu parādīšanās biežumu.

---

## 🔍 Analīze

### Problēmas apraksts

Tradicionāla skolēnu uzskaite (piemēram, klases žurnālā vai manuāli atzīmējot apmeklējumu) ir lēna, pakļauta cilvēciskām kļūdām un laikietilpīga. Automatizēta seju atpazīšana ļauj paātrināt procesu, mazināt kļūdas un savākt papildus statistiku.

### Mērķauditorija

- Skolu administrācija un skolotāji
- Tehnoloģiju skolas vai fakultātes
- Programmētāji, kuri vēlas paplašināt zināšanas par datorredzi

### Eksistējošo risinājumu analīze

| Nosaukums | Apraksts | Plusi | Mīnusi |
|----------|----------|--------|--------|
| OpenCV Attendance System (GitHub) | Vienkāršs skripts apmeklējuma reģistrēšanai | Viegli saprotams, labs pamats | Nav statistikas, nav verifikācijas |
| Commercial solutions (FaceFirst, Trueface) | Profesionālas sistēmas | Precīzas, drošas | Maksas, nav atvērtā koda, nav pielāgojamas |
| Paštaisīti CSV + kamera | Minimāla seju detekcija | Vienkārši | Nav atpazīšanas, tikai detekcija |

📸 *(Ekrānšāviņus vari pievienot vēlāk ar reāliem piemēriem no sava koda!)*

---

## 🧩 Projektēšana

### Funkcionālās prasības
1. Atpazīt skolēnu sejas no attēliem vai video
2. Glabāt skolēnu ID, vārdu un sejas kodējumu datubāzē (CSV)
3. Pierakstīt katru parādīšanās reizi ar laiku
4. Brīdināt, ja seja nav atpazīta (nezināms students)
5. Ļaut pievienot jaunu studentu datubāzei

### Nefunkcionālās prasības
1. Lietotnei jābūt izpildāmai no konsoles (CLI)
2. Darbībai jābūt iespējamai bez interneta
3. Jāstrādā ar attēlu vai video failiem (ne obligāti reāllaikā)
4. Lietotāja interfeisam jābūt vienkāršam un saprotamam
5. Datu glabāšanai jābūt drošai (nav piekļuves trešajām pusēm)

---

## 🗓️ Plānošana – darba uzdevumu saraksts

1. Izveidot `Student` klasi ar ID, vārdu un sejas enkodējumu
2. Realizēt CSV datubāzes lasīšanu un rakstīšanu
3. Implementēt sejas atpazīšanu ar `face_recognition`
4. Saglabāt notikumu žurnālu ar datumu/laiku
5. Izveidot CLI izvēlni (atpazīšana / pievienošana / statistika)

---

## 🎥 Risinājuma prezentācija

🧪 Pievienotie ekrānšāviņi:

- ✅ Seja atpazīta → uz ekrāna parādās vārds un laiks
- ⚠️ Nezināma seja → brīdinājums un piedāvājums pievienot
- 📈 Statistikas CSV fails ar skolēna vārdu, datumu un reižu skaitu

*(Ekrānšāviņus vari augšupielādēt savā GitHub repozitorijā mapē `/screenshots/`)*

---

## 💻 Tehnoloģijas

- Python 3.x
- face_recognition
- OpenCV
- Pandas

---

## 📁 Struktūra (piemērs)

FaceTrackEd/
├── main.py                       # Точка входа в программу
├── requirements.txt              # Зависимости проекта
├── README.md                     # Документация проекта
├── config/
│   └── settings.json             # Настройки в формате JSON
│   └── settings.py               # Класс загрузки настроек
├── data/
│   ├── students.csv              # База данных студентов
│   ├── log.csv                   # Лог посещений
│   └── faces/                    # Папка с изображениями лиц студентов
├── app/
│   ├── __init__.py
│   ├── database/
│   │   ├── __init__.py
│   │   ├── db.py                 # Классы: StudentDatabase, AttendanceLogger
│   ├── face/
│   │   ├── __init__.py
│   │   ├── recognition.py        # Класс: FaceRecognizer
│   ├── stats/
│   │   ├── __init__.py
│   │   ├── analytics.py          # Класс: AttendanceStats
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── helpers.py            # Класс: TimeUtils и прочие утилиты
│   └── core/
│       ├── __init__.py
│       ├── app.py                # Класс: FaceTrackApp — главный контроллер

# Пояснение по файлам:
# - main.py — запускает приложение, вызывает FaceTrackApp.run()
# - settings.json — хранит пути, настройки камеры, формат даты и т.д.
# - db.py — управляет загрузкой/сохранением студентов и логов
# - recognition.py — кодирует и распознает лица с камеры
# - analytics.py — обрабатывает и отображает статистику
# - app.py — объединяет всё в единое приложение
# - helpers.py — содержит утилиты: таймстемпы, работа с путями и т.д.


## ✅ To-Do

### 🔹 Projekta sākums
- [x] Izveidot GitHub repozitoriju: `FaceTrackEd`
- [x] Izveidot `README.md` ar problēmas aprakstu, mērķiem un plānu
- [ ] Sagatavot testēšanas attēlus (studentu sejas)

---

### 🔹 Datu struktūras un klase
- [x] Izveidot `Student` klasi ar laukiem: `id`, `name`, `encoding`
- [x] Izveidot CSV failu `students.csv`, kur glabāt datus
- [x] Pievienot iespēju saglabāt / nolasīt `encoding` sarakstu no CSV

---

### 🔹 Seju atpazīšanas funkcionalitāte
- [x] Ielādēt attēlu vai video
- [x] Atpazīt sejas attēlā ar `face_recognition`
- [x] Salīdzināt ar esošajām sejām datubāzē
- [x] Ja seja neatpazīta – brīdināt un piedāvāt pievienot

---

### 🔹 Statistikas funkcijas
- [x] Saglabāt katru atpazīšanu `log.csv` ar laiku un skolēna ID
- [x] Iegūt statistiku par parādīšanās biežumu
- [ ] Izveidot grafikus ar `Matplotlib` (papildus iespēja)

---

### 🔹 Lietotāja interfeiss (CLI)
- [x] Izveidot vienkāršu izvēlni terminālī:
  - [x] ✅ Atpazīt seju
  - [x] ➕ Pievienot jaunu studentu
  - [x] 📊 Apskatīt statistiku
- [x] Validēt ievadītos datus (piemēram, vārda ievade)

---

### 🔹 Testēšana
- [ ] Testēt ar vairākām sejām un attēliem
- [ ] Simulēt kļūdas (piemēram, nav encodinga)
- [ ] Testēt CSV failu bojājumus un atkopšanu

---

### 🔹 Prezentācijai
- [ ] Uztaisīt ekrānšāviņus: atpazīšana, kļūda, pievienošana
- [ ] Aprakstīt projekta gaitu prezentācijā (PowerPoint vai PDF)
- [ ] Izveidot demonstrācijas video vai ekrānuzņēmumu GIF

