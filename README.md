# Secure MOM — asistent AI on-premise pentru procese-verbale multilingve

**DeepTech GigaHack 2026 · Challenge Medpark International Hospital**

Secure MOM primește înregistrarea unei ședințe (audio sau video, încărcată prin drag & drop sau înregistrată direct din browser cu „Rec”). Din ea produce un **proces-verbal structurat**: decizii, sarcini, responsabili, termene și participanți. Documentul e trimis automat pe email listei de distribuție a tipului de ședință. **Totul rulează local, fără nicio conexiune la internet.**

```
audio/video ──► curățare zgomot ──► ASR RO/RU/EN ──────────► diarizare ──► LLM local ──► DOCX/HTML ──► n8n ──► email intern
  (upload/Rec)    ffmpeg+spectral     RO: Whisper moldovenesc   ECAPA +       qwen3.5:9b    RO/RU/EN      rutare     Mailpit
                  gating              RU/EN: Whisper multiling. spectral      (Ollama)                    pe tip     (SMTP local)
                                      + test lexical + vocabular clustering
                                        medical (SiMoNERo/MoNERo)
```

## Pornire

```bash
./scripts/setup.sh     # O SINGURĂ DATĂ, cu internet: dependențe + modele open-weights (Whisper, ECAPA, qwen3.5:9b, n8n)
./scripts/start.sh     # apoi: deconectați rețeaua. Pornește Mailpit + Ollama + n8n + serverul
```

| Serviciu | Adresă (doar 127.0.0.1) |
|---|---|
| Interfața Secure MOM | http://127.0.0.1:8000 |
| Căsuța de email internă (Mailpit) | http://127.0.0.1:8025 |
| n8n (workflow de rutare) | http://127.0.0.1:5678 |
| API (OpenAPI) | http://127.0.0.1:8000/api/docs |

Rulare fără interfață: `python -m scripts.run_cli inregistrare.m4a --type medical --lang ro`.

## Cum răspunde soluția la criteriile de jurizare

### 1. Securitate și arhitectură: zero apeluri externe

- **Toate modelele rulează local:** Whisper (MLX pe Apple Silicon, faster-whisper/CTranslate2 pe server CUDA sau CPU), ECAPA-TDNN (SpeechBrain) și qwen3.5:9b (Ollama). Emailul trece prin n8n self-hosted și Mailpit.
- **Garda offline** ([app/offline_guard.py](app/offline_guard.py)). La pornirea procesului înlocuiește `socket.connect` și `getaddrinfo`: orice conexiune spre o adresă din afara rețelei interne (`127/8`, `10/8`, `172.16/12`, `192.168/16`) e **refuzată și înregistrată**. Tentativele blocate apar live în interfață (indicatorul „Offline guard”) și în `/api/health`.
- `HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE` și `DO_NOT_TRACK` sunt forțate. SpeechBrain e obligat să citească ponderile de pe disc. n8n rulează cu telemetria, șabloanele, notificările de versiune și funcțiile AI dezactivate.
- **Datele** (audio, transcrieri, rapoarte) rămân în `data/jobs/<id>/` pe server și pot fi șterse din API (`DELETE /api/jobs/{id}`).
- **Hardware de referință:** un singur GPU de 16 GB sau un server doar-CPU. Joburile rulează secvențial, iar Whisper e eliberat din memorie înainte de pornirea LLM-ului. Vârful de memorie e dat de LLM (~6 GB), la care se adaugă Whisper large-v3 (~3 GB) numai în timpul etapei ASR. Pe un server fără Apple Silicon, backend-ul se alege automat: faster-whisper cu `float16` pe CUDA sau `int8` pe CPU.

### 2. Acuratețe lingvistică: code-switching RO/RU/EN și dialectul moldovenesc

Toate cifrele de mai jos sunt **măsurate**, cu scripturi reproductibile din `training/`.

**Arhitectura cu două modele Whisper** ([app/asr.py](app/asr.py)):

| Rol | Model | De ce |
|---|---|---|
| Română (limba de stat) | [FraPiz/whisper-large-v3-turbo-moldovan-romanian](https://huggingface.co/FraPiz/whisper-large-v3-turbo-moldovan-romanian) (Apache 2.0), antrenat pe cele **70 h** ale corpusului de vorbire moldovenească, **apoi adaptat de noi pe terminologia medicală** (vezi mai jos) | cel mai bun pe accentul, regionalismele moldovenești și termenii medicali |
| Detecția limbii, rusă, engleză | Whisper large-v3-turbo multilingv | cel mai bun pe code-switching (7/8 limbi corecte) |

Două descoperiri au dus la această arhitectură ([training/eval_codeswitch.py](training/eval_codeswitch.py), test sintetic RO/RU/EN generat offline cu vocile macOS):
- modelul adaptat pe dialect **„uită” rusa**: o scrie cu litere românești („hărășo, ia zdela…”);
- Whisper large-v3, forțat pe română, **traduce** rusa și engleza în română („Bine, voi face tomografia…”). Textul arată corect, dar nu e ce s-a spus. De aceea a fost scos din sistem.

**Decizia de limbă, frază cu frază.** Ipoteza română vine de la specialist, ipotezele rusă/engleză de la modelul multilingv. Câștigă cea cu `log-prob + 2 × (proporția de cuvinte reale ale limbii) + prior`. Rusa transliterată sau textul cu alfabet amestecat e penalizat. O ipoteză cu prea puține cuvinte pentru durata audio (specialistul se oprește uneori când fraza trece în engleză sau rusă) nu e acceptată direct: se cere și opinia modelului multilingv, iar varianta trunchiată e penalizată. Lexiconul fiecărei limbi e construit din:
- 50k cuvinte frecvente;
- vocabularul corpusului moldovenesc (20.236 de forme);
- **vocabularul medical** din [SiMoNERo/MoNERo](https://github.com/UniversalDependencies/UD_Romanian-SiMoNERo): 16.747 de forme, 7.313 termeni ANAT/CHEM/DISO/PROC;
- vocabularul medical rusesc din [rus_med_dialogues](https://huggingface.co/datasets/Mykes/rus_med_dialogues) și [medical_qa_ru_data](https://huggingface.co/datasets/blinoff/medical_qa_ru_data): 17.304 termeni.

**Terminologie medicală:**
- promptul Whisper e orientat pe tipul ședinței, cu termeni din glosar;
- glosarul corectează erorile observate („miropinem” → meropenem);
- o **corectare fonetică deterministă** ([app/medcorrect.py](app/medcorrect.py)) restaurează diacriticele lipsă și înlocuiește *doar cuvintele inexistente* cu termenul medical care sună aproape identic, după o cheie fonetică adaptată pronunției moldovenești („nevostomă” → nefrostomă);
- numeralele devin cifre ([app/ronum.py](app/ronum.py)): „zero virgulă douăzeci și doi” → „0,22”, „trei opt patruzeci” → „38 40” (nu se adună). „Pacientul patru opt”, adică „pa**tu'** opt” în pronunția moldovenească, devine „patul 8”, nu „48”: în română 48 se rostește „patruzeci și opt”.

**Antrenare locală pe terminologia medicală** ([training/finetune_decoder_mlx.py](training/finetune_decoder_mlx.py)). Am adaptat modelul moldovenesc pe GPU-ul laptopului (MLX), fără niciun serviciu extern:
- **date:** 2.500 de fraze medicale românești (SiMoNERo/MoNERo + glosar în șabloane de consiliu medical), citite de vocea TTS locală și „murdărite” cu zgomot de sală și reverberație ([training/make_tts_medical.py](training/make_tts_medical.py)), amestecate cu 1.000 de clipuri **reale** din corpusul moldovenesc, ca modelul să nu uite dialectul;
- **ce se antrenează:** doar blocurile decoderului (105 M parametri, partea de „limbaj”). Encoderul, adică „urechea” adaptată pe accentul moldovenesc, rămâne înghețat;
- **durată:** 2 epoci, 874 de pași, ~45 min pe MacBook M5 16 GB (vârf de memorie 7 GB);
- **server:** modelul e exportat și pentru faster-whisper/CUDA ([training/convert_mlx_to_ct2.py](training/convert_mlx_to_ct2.py)). Dacă modelul adaptat lipsește, aplicația folosește automat modelul moldovenesc de bază.

Transcrierea de referință Medpark **nu** a fost folosită la antrenare, doar la evaluare.

**Runda a doua: voci moldovenești reale, code-switching în frază, „sala de consiliu”** ([training/finetune_whisper_mlx.py](training/finetune_whisper_mlx.py)). Tot local, pe laptop:
- **texte:** 1.607 replici de ședință în stil vorbit moldovenesc, generate de LLM-ul local ([training/make_medical_texts.py](training/make_medical_texts.py)). Numerele sunt rostite în litere, iar ~40% din replici conțin rusă **în interiorul frazei** („…давление упало, trebuie să punem stent”), ~20% engleză;
- **voci:** [FraPiz/xtts-v2-moldovan-romanian](https://huggingface.co/FraPiz/xtts-v2-moldovan-romanian) (XTTS-v2 adaptat pe 44 h de română moldovenească) clonează vocile a **48 de profesori reali** din corpus. Modelul moldovenesc „a uitat” rusa (măsurat), așa că bucățile rusești și englezești sunt rostite de XTTS-v2 original, **cu aceeași voce** ([training/make_tts_xtts.py](training/make_tts_xtts.py));
- **verificare dus-întors:** fiecare bucată e retranscrisă în limba ei. Clipurile în care sinteza a sărit sau a inventat cuvinte sunt eliminate (1.146 păstrate din 1.281) ([training/filter_tts.py](training/filter_tts.py)). Altfel ASR-ul ar învăța să halucineze;
- **sala:** la fiecare pas, fiecare clip trece printr-o sală aleatoare: reverberație, 1–3 voci reale care vorbesc pe fundal, zgomot de aparatură, bandă de telefon, volum variabil;
- **ce se antrenează:** decoderul și ultimele 4 straturi ale encoderului (184 M parametri). 1.498 de clipuri moldovenești reale sunt amestecate ca „reamintire” a dialectului. 3 epoci, 107 min pe MacBook M5;
- **WiSE-FT** (Wortsman et al., 2022): modelul final e media ponderilor 30% runda 1 + 70% runda 2. Singur, modelul din runda 2 câștiga mult pe code-switching, dar pe înregistrarea reală inventa uneori rusă sau medicamente („ceftarabia”). Media păstrează aproape tot câștigul, fără aceste invenții.

**Test de generalizare cu vorbitori NEVĂZUȚI** ([training/eval_synth.py](training/eval_synth.py)): 50 de replici scoase din antrenare, rostite de vocile a 5 profesori care nu apar nicăieri în antrenare, trecute prin tot sistemul:

| Specialist RO | Curat: WER latin / WER rusă / numere / termeni critici | Sală: WER latin / numere / termeni critici |
|---|---|---|
| runda 1 | 38.7% / 109% / 88% / 57% | 58.5% / 72% / 47% |
| **runda 2 + WiSE-FT (folosit)** | **23.9% / 52% / 88% / 68%** | **42.7% / 80% / 57%** |

„WER rusă” peste 100% la runda 1 înseamnă că rusa din mijlocul frazei românești era scrisă cu litere latine sau omisă. Acum e scrisă în chirilică.

**Rezultate pe înregistrarea reală Medpark**, față de transcrierea manuală a echipei (două ferestre, 2 min 36 s, [data/reference/medpark_ref.json](data/reference/medpark_ref.json), [training/eval_reference.py](training/eval_reference.py)):

| Sistem | WER | CER | Termeni medicali recunoscuți |
|---|---|---|---|
| Whisper large-v3, segmente de 12 s (versiunea inițială) | 64.4% | 42.3% | 64% |
| large-v3, segmente de 25 s (mai mult context) | 56.4% | 36.4% | 64% |
| Două modele (moldovenesc + multilingv) + corectare fonetică | 51.9% | 31.0% | 73% |
| + modelul moldovenesc adaptat de noi pe terminologia medicală (runda 1) + corecturi numerale | 45.2% | 25.2% | 82% |
| **+ runda 2 (voci moldovenești, code-switching, sală) cu WiSE-FT — folosit** | 47.7% | 25.3% | 77% |

Înregistrarea e făcută cu un singur telefon în sală, cu mai mulți vorbitori care se suprapun și fragmente pe care nici omul nu le înțelege. Eroarea absolută rămâne mare, dar a scăzut cu **30% (WER) și 39% (CER)** față de versiunea inițială.

**Costul adaptării:** pe 150 de clipuri din corpusul moldovenesc general (non-medical), WER a crescut în runda 1 de la 3.2% la 4.3%. Runda 2, care folosește replay cu clipuri reale trecute prin „sală”, a coborât la 3.8%.

**Ce arată onest cifrele:** pe cele 2,6 minute de referință Medpark, runda 2 are CER egal (25.3% față de 25.2%) și WER cu 2,5 puncte mai mare, adică ~10 cuvinte. Pe testul cu vorbitori nevăzuți câștigă 15–57 de puncte. Am ales varianta care generalizează: juriul va testa pe o înregistrare nouă, iar code-switching-ul e miezul provocării. Modelul din runda 1 rămâne automat ca rezervă (`ro_mlx_model` e o listă în [config/config.yaml](config/config.yaml)).
*Limitare:* exportul CTranslate2 al modelului din runda 2 nu a trecut verificarea. Pe serverul CUDA se folosește deocamdată modelul din runda 1 (export verificat).

**Ce am încercat și am respins pe baza măsurătorilor:**
- **corectarea transcrierii cu LLM-ul:** WER 65.8% → 66.1%, termeni 64% → 59%; LLM-ul „corectează” și forme deja bune. Codul rămâne pentru experimente, dezactivat.
- **large-v3 ca model principal:** traduce în loc să transcrie.

**Test sintetic de code-switching** (10 fraze RO/RU/EN, 2 cu schimbare de limbă în mijloc):

| Sistem | WER RO | WER RU | WER EN | Limbă corectă |
|---|---|---|---|---|
| large-v3 | 6% | 116% (traduce) | 100% (traduce) | 4/8 |
| model moldovenesc singur | 12% | 125% (transliterează) | 13% | 3/8 |
| două modele | 12% | 38% | 13% | 6/8 |
| **două modele, specialist adaptat medical + test de acoperire** | **6%** | **22%** | **7%** | **7/8** |

### 3. Calitatea procesului-verbal

LLM-ul local ([app/llm.py](app/llm.py)) primește transcrierea cu marcaje de timp, vorbitori și limbi. Produce JSON strict (schemă forțată), în limba de stat:

- **decizii** (ce s-a hotărât, de cine, despre ce pacient sau subiect);
- **sarcini** (responsabil, termen, prioritate);
- participanți, subiecte, probleme deschise, ședința următoare.

**Garanții împotriva invențiilor:**
- Fiecare decizie sau sarcină are un **citat-dovadă copiat din transcriere**, verificat fuzzy în Python. Ce nu se regăsește în transcriere e eliminat; ce se potrivește parțial e marcat „de verificat”.
- **Termenele** se trec doar dacă au fost rostite. Termenele relative („până vineri”) sunt convertite la dată calendaristică față de data ședinței și validate. Termenele „deduse” sau „standard” sunt respinse automat.
- Pentru ședințele lungi se aplică map-reduce: extragerea pe bucăți, apoi **deduplicarea deterministă** și o sinteză scurtă.
- Numele participanților apar doar dacă sunt rostite în ședință („am chemat deja pe Butnari”).
- **Cuvintele critice nesigure sunt semnalate, nu ascunse.** După transcriere, o singură trecere suplimentară a decoderului dă probabilitatea fiecărui cuvânt ([app/asr.py](app/asr.py)). Probabilitatea se păstrează și după corecturi: „patru opt” → „patul 8” rămâne nesigur. Dacă o decizie sau o sarcină se sprijină pe un **număr, o doză, un medicament sau o procedură** recunoscut cu încredere scăzută, documentul afișează „⚠ verificați la audio: 8 (31%)”. În transcrierea din interfață aceste cuvinte sunt marcate cu roșu, iar un click pe ora frazei redă fragmentul audio. Pe o înregistrare de consiliu făcută cu un telefon, niciun ASR nu e perfect. Sistemul arată exact unde trebuie să se uite omul.

**Conținutul documentului** ([app/report.py](app/report.py)) urmează strict cerința challenge-ului. Documentul DOCX și HTML (HTML-ul e și corpul emailului) conține:
1. **Antet:** tipul ședinței, data, durata și participanții.
2. **Rezumat** al ședinței.
3. **Decizii**, grupate pe pacient („Pacient — Patul 8”) la consiliul medical. Gruparea e deterministă, după numărul patului.
4. **Sarcini (action items)**, cu coloanele ☐ · Sarcina · **Responsabil** · **Termen**. Dacă termenul nu a fost rostit, apare „nestabilit”.

Transcrierea **nu** intră în document. Formatarea clinică ([app/clinical.py](app/clinical.py)) folosește abrevieri standard (TA, SpO2, NA, FEVS…). Bold primesc doar valorile anormale și dozele. Speculațiile LLM-ului („(probabil …)”) sunt eliminate. Blocul „⚠️ Necesită verificare umană” se poate reactiva cu `output.review_block: true`. **Limba documentului** (RO/RU/EN) se alege înainte sau după procesare. Traducerea e făcută local.

### 4. Experiența utilizatorului

Interfața e gândită pentru medici: minimalistă, fără detalii tehnice. Medicul face trei pași:
1. trage fișierul în pagină sau apasă **Înregistrați ședința**;
2. alege tipul ședinței și limba documentului;
3. apasă **Generează procesul-verbal**.

În timpul procesării apare o singură bară de progres (0–100%) și timpul scurs, fără etapele interne. La final:
- procesul-verbal pleacă **automat** la lista de distribuție a tipului de ședință (n8n, `delivery.auto_send`);
- pe pagina de rezultat, **Trimite pe email** cere adresa și trimite documentul doar acolo;
- **Descarcă Word** salvează fișierul DOCX.

Istoricul ședințelor e pe o pagină separată.

Timp măsurat end-to-end (upload → email în căsuță), MacBook M5 16 GB, înregistrarea Medpark de 11 min 42 s:

| Etapă | Timp |
|---|---|
| Curățare audio (ffmpeg + spectral gating) | 2 s |
| ASR: Whisper moldovenesc (runda 2) + multilingv + încredere pe cuvânt | 71 s |
| Diarizare (GPU) | 6 s |
| LLM (qwen3.5:9b, determinist) | 103 s |
| Document + n8n + email | < 1 s |
| **Total (măsurat prin API, upload → email în Mailpit)** | **3 min 07 s** |

Extrapolat la **60 de minute**: ~15 min pe acest laptop. Pe serverul de referință cu GPU CUDA de 16 GB, faster-whisper e sensibil mai rapid decât MLX pe laptop. LLM-ul rulează cu temperatură 0 și seed fix: **același audio produce mereu același proces-verbal** (verificat).

### 5. Automatizare și rutare (n8n)

Workflow-ul [n8n/workflow.json](n8n/workflow.json) e importat și activat automat de `start.sh`. Fluxul:

**Webhook** → **Switch „Tip ședință”** (medical / executive / administrative) → ramura dedicată (lista de distribuție, prioritatea, atașamentul DOCX) → **Send Email** prin SMTP-ul intern → răspuns cu ruta aleasă.

Dacă n8n nu răspunde, serverul trimite direct prin SMTP-ul local, ca procesul-verbal să ajungă oricum. Listele de distribuție se configurează în [config/config.yaml](config/config.yaml).

## Zgomot de fond

În [app/audio.py](app/audio.py), două variante ale semnalului sunt produse dintr-o singură decodare:

- **pentru ASR:** filtre de bandă (80 Hz – 7.8 kHz, taie vibrațiile, pașii, hum-ul și șuieratul) și normalizare dinamică, ca vorbitorii departe de microfon să ajungă la nivelul celor apropiați. Whisper e antrenat pe audio zgomotos, iar denoise-ul agresiv îi strică detecția de limbă (măsurat).
- **pentru VAD și diarizare:** în plus, denoise FFT pentru zgomotul staționar (ventilație, aparatură) și spectral gating non-staționar pentru foșnet de hârtie, pocnituri și voci din fundal.

Pe înregistrarea Medpark, zgomotul de fond scade de la −32.8 dB la −43.6 dB.

## Participanți (diarizare)

În [app/diarize.py](app/diarize.py): amprente vocale ECAPA-TDNN pe ferestre de 1.5 s, apoi clustering spectral. **Numărul de vorbitori nu e fixat**: e estimat din date (eigengap NME-SC). Frazele în care vorbitorul se schimbă sunt împărțite. Opțional, utilizatorul poate indica numărul de participanți dacă îl cunoaște.

*Limitare onestă:* pe o înregistrare făcută cu un singur telefon în sală, amprentele vocale separă slab persoanele aflate la distanță similară de microfon. Pentru producție recomandăm un microfon de conferință.

## Nimic hardcodat

Modelele, limbile, pragurile, profilurile ASR, tipurile de ședință, listele de distribuție, glosarul medical și lexicoanele stau în [config/config.yaml](config/config.yaml) și [config/medical_glossary.yaml](config/medical_glossary.yaml). Orice valoare se poate suprascrie cu variabile de mediu `SECURE_MOM_<SECȚIUNE>__<CHEIE>`. Numărul de participanți, limbile vorbite, deciziile și termenele rezultă din înregistrare.

## Structura proiectului

```
app/          audio.py · asr.py · lexicon.py · glossary.py · medcorrect.py · ronum.py · diarize.py · llm.py
              report.py · delivery.py
              pipeline.py (orchestrare + coadă) · main.py (FastAPI) · offline_guard.py · config.py
web/          interfața (HTML/CSS/JS, fără CDN-uri — funcționează offline)
config/       config.yaml · medical_glossary.yaml
n8n/          workflow.json · credentials.json (SMTP intern)
training/     build_lexicon.py (vocabular moldovenesc) · build_medical_vocab.py (SiMoNERo/MoNERo, medical RU)
              convert_hf_to_mlx.py (Whisper moldovenesc -> MLX) · eval_reference.py (WER pe Medpark vs referință)
              make_tts_medical.py (fraze medicale sintetice) · finetune_decoder_mlx.py (adaptare medicală)
              convert_mlx_to_ct2.py (export pentru faster-whisper)
              make_medical_texts.py · make_tts_xtts.py · filter_tts.py · finetune_whisper_mlx.py (runda 2)
              eval_synth.py (test cu vorbitori nevăzuți)
              eval_codeswitch.py (test RO/RU/EN) · eval_wer.py (WER pe corpusul moldovenesc)
scripts/      setup.sh · start.sh · run_cli.py
```
# CodeRiders
