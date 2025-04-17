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

FaceTrackEd/ ├── main.py ├── students.csv ├── log.csv ├── data/ │ └── atteli/ └── screenshots/


###✅ FaceTrackEd – ToDo saraksts
🔹 1. Projekta sākums

Izveidot GitHub repozitoriju: FaceTrackEd

Izveidot README.md ar problēmas aprakstu, mērķiem un plānu

    Sagatavot testēšanas attēlus (studentu sejas)

🔹 2. Datu struktūras un klase

Izveidot Student klasi ar laukiem: id, name, encoding

Izveidot CSV failu students.csv, kur glabāt datus

    Pievienot iespēju saglabāt / nolasīt encoding sarakstu no CSV

🔹 3. Seju atpazīšanas funkcionalitāte

Ielādēt attēlu vai video

Atpazīt sejas attēlā ar face_recognition

Salīdzināt ar esošajām sejām datubāzē

    Ja seja neatpazīta – brīdināt un piedāvāt pievienot

🔹 4. Statistikas funkcijas

Saglabāt katru atpazīšanu log.csv ar laiku un skolēna ID

Iegūt statistiku par parādīšanās biežumu

    Izveidot grafikus ar Matplotlib (papildus)

🔹 5. Lietotāja interfeiss

Izveidot vienkāršu izvēlni terminālī:

    ✅ Atpazīt seju

    ➕ Pievienot jaunu studentu

    📊 Apskatīt statistiku

    Pievienot validācijas (piemēram, vārda ievade)

🔹 6. Testēšana

Testēt ar vairākām sejām

Simulēt kļūdas (piemēram, nav encodinga)

    Testēt CSV failu bojājumus / nolasīšanu

🔹 7. Prezentācijai

Uztaisīt ekrānšāviņus katram posmam (atpazīšana, kļūda, pievienošana)

Aprakstīt projekta norisi prezentācijā

    Izveidot demonstrācijas video vai īsu demo failu

Gribi, lai arī sadalu šo pa nedēļām kā plānu uz 1,5 mēnešiem? Tas varētu palīdzēt prezentācijā vai plānošanā.

