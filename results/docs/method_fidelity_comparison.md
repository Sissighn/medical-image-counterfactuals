# Originaltreue der vier Counterfactual-Methoden — ausführlicher Vergleich

Dieses Dokument hält für jede der vier implementierten Counterfactual-Methoden
fest, **was originalgetreu** umgesetzt ist, **worin sie sich vom Original
unterscheiden**, und **welche bewussten Design-Entscheidungen** getroffen
wurden. Grundlage ist ein Zeile-für-Zeile-Abgleich der Projekt-Implementierungen
gegen die jeweiligen Originalquellen.

Framework- und Klassifikator-Unterschiede (TensorFlow → PyTorch, ImageNet-CNN →
medizinisches ResNet-18) sind projektweit gewollt und werden nicht als Fehler
gewertet.

## Referenzen

| Methode | Projekt-Implementierung | Originalquelle |
| --- | --- | --- |
| CFProto | [`scripts/run_cfproto_pytorch.py`](../../scripts/run_cfproto_pytorch.py) | Alibi `CounterfactualProto` — <https://github.com/SeldonIO/alibi> (`alibi/explainers/cfproto.py`) |
| Goyal et al. CVE | [`scripts/run_goyal_cve_pytorch.py`](../../scripts/run_goyal_cve_pytorch.py) | Goyal et al. 2019 (arXiv:1904.07451); Referenzcode <https://github.com/facebookresearch/visual-counterfactuals> |
| SEDC-T | [`scripts/run_sedc_t_pytorch.py`](../../scripts/run_sedc_t_pytorch.py) | `sedc_t2_fast.py` — <https://github.com/ADMAntwerp/ImageCounterfactualExplanations> |
| DVCE | [`src/dvce_core.py`](../../src/dvce_core.py), [`scripts/run_dvce_pytorch.py`](../../scripts/run_dvce_pytorch.py) | DVCEs — <https://github.com/valentyn1boreiko/DVCEs> (vendored unter [`external/DVCEs`](../../external/DVCEs)) |

**Gesamtergebnis:** Alle vier Kernalgorithmen sind exakt portiert. Es wurde kein
methodischer Fehler gefunden. Sämtliche Abweichungen sind bewusste, im Code
dokumentierte Design-Entscheidungen (Framework, Datensatz, Evaluationsprotokoll)
und ändern den Algorithmus selbst nicht.

---

## 1. CFProto (prototyp-geführte Counterfactuals)

Portierung des TensorFlow-1.x-Graphen von Alibi `CounterfactualProto.attack`
nach PyTorch, angewendet auf ein medizinisches ResNet-18 mit einem
Faltungs-Autoencoder als Encoder.

### 1.1 Was originalgetreu ist

- **Attack-Loss (Hinge):** `f(x,d) = max(0, p_orig − max_{i≠orig} p_i + kappa)`
  inklusive des `−10000`-Maskierungstricks zum Ausschluss der Originalklasse aus
  dem Nicht-Ziel-Maximum. Entspricht exakt Alibis `loss_attack`
  ([`run_cfproto_pytorch.py:232-240`](../../scripts/run_cfproto_pytorch.py#L232-L240)).
- **Optimierte Loss:** `c·L_attack + L2 + gamma·L_AE + theta·L_proto`, wobei der
  `beta·L1`-Term **nicht** über den Gradienten, sondern über das
  Shrinkage-Thresholding eingebracht wird. Alle Terme sind **Summen** (nicht
  Mittelwerte), exakt wie im TF-Graphen (`loss_opt`).
- **FISTA-Optimierung:** Element-weises Shrinkage-Thresholding mit identischer
  Drei-Fall-Struktur (`delta > beta` → oberer Zweig, `|delta| ≤ beta` → zurück
  auf das Original, sonst unterer Zweig) und Projektion auf die Feature-Range.
  Nesterov-Momentum `zt = (i+1)/(i+4)` (entspricht Alibis
  `global_step/(global_step+3)` nach dem Inkrement), abschließendes Clamp auf
  `(0, 1)` ([`run_cfproto_pytorch.py:243-253`, `348-358`](../../scripts/run_cfproto_pytorch.py#L243-L253)).
- **Lernraten-Decay:** Polynomialer Zerfall mit Power 0.5 auf Endwert 0 — exakt
  Alibis `tf.train.polynomial_decay(..., power=0.5)`.
- **Gradient-Clipping:** auf `(−1000, 1000)`, der Alibi-Default `clip`.
- **Binäre Suche über `c`:** Obergrenze `ub = min(ub, c)`, Bisektion sobald
  `ub < 1e9`, sonst Verzehnfachung von `c` — identisch zu Alibis
  Konstanten-Update. `adv` und `adv_s` werden pro c-Step auf das Original
  reinitialisiert.
- **Prototypen-Bildung (`fit`):** Klassenzugehörigkeit über die
  **Klassifikator-Vorhersagen** auf den Trainingsdaten, **nicht** über die
  wahren Labels (`preds = argmax(predict(train_data))`). Encoder-Raum-Prototypen
  als Klassenmittel bzw. k-nächste-Nachbarn.
- **kNN-Prototyp-Semantik:** Bei gesetztem `k` liefert `k_type='mean'` die
  mittlere Distanz und `k_type='point'` die Distanz des k-ten Nachbarn — die
  Prototyp-Position ist in **beiden** Fällen das Mittel der k nächsten
  Encodings. Das repliziert Alibis (leicht kontraintuitive) Implementierung
  exakt ([`run_cfproto_pytorch.py:188-205`](../../scripts/run_cfproto_pytorch.py#L188-L205)
  vs. Alibi `cfproto.py:1043-1051`).
- **Nächster-Prototyp-Zielwahl:** Unter allen Kandidatenklassen wird diejenige
  mit minimaler Distanz `ENC(x)`→Prototyp gewählt (`id_proto = min(dist_proto)`).
- **Best-Counterfactual-Auswahl:** über die Elastic-Net-Distanz `L2 + beta·L1`
  in Kombination mit der kappa-Bedingung `compare()` und der
  Zielklassen-Mitgliedschaft.

### 1.2 Unterschiede zum Original

- **Framework:** PyTorch-Reimplementierung statt des TF-1.x-Graphen. Der
  Klassifikator ist ein ResNet-18, dessen ImageNet-Normalisierung in die
  `predict`-Funktion gewickelt ist; die Optimierung läuft im `[0,1]`-Pixelraum,
  passend zu `feature_range=(0,1)`.
- **Kein Black-Box-Modus:** Der numerische Gradientenpfad des Originals (für
  nicht-differenzierbare Modelle) ist weggelassen — hier ist der Klassifikator
  immer differenzierbar.
- **Keine kategorialen Variablen / k-d-Bäume:** Die ABDM/MVDM-Distanzen und die
  k-d-Baum-Prototypen des Originals entfallen (nur für tabellarische/kategoriale
  Daten relevant, hier sind es Bilder).
- **Kein TrustScore-Filter:** Der optionale TrustScore-Schwellwert ist nicht
  implementiert — der Alibi-Default `threshold=0` deaktiviert ihn ohnehin, das
  Verhalten ist also identisch.

### 1.3 Bewusste Design-Entscheidungen

- **Hyperparameter nach dem Alibi-MNIST-Beispiel statt den Klassen-Defaults:**
  `c_init=1, c_steps=2` (Alibi-Klassendefault wäre 10 / 10). Bewusst gewählt, um
  die Laufzeit auf den medizinischen Bildern beherrschbar zu halten; die
  Bisektions-Logik ist identisch.
- **Reskalierung von `gamma` und `theta`:** Die Alibi-MNIST-Beispiele nutzen 100
  auf 28×28-Eingaben bzw. einem kleinen Latent-Raum. Da alle Loss-Terme Summen
  über die Dimensionen sind, müssen die Gewichte auf 224×224 bzw. die
  Encoder-Dimensionalität umskaliert werden, damit die Terme vergleichbar
  bleiben. Defaults hier: `gamma=1.0`, `theta=0.5` (im Argument-Help begründet,
  Kontrolle über `loss_terms` in `metadata.json`).
- **Verhalten bei „kein CF gefunden":** Statt Alibis Null-Array wird das letzte
  `adv` mit `found=False`/`valid=False` zurückgegeben. Rein für das Reporting
  sauberer; die Validitäts-Statistik zählt es korrekt als ungültig.
- **Encoder-Raum-Prototypen:** Es wird durchgängig ein Autoencoder-Encoder
  genutzt (kein k-d-Baum-Fallback), was Alibis `enc_model`-Pfad entspricht.

---

## 2. Goyal et al. 2019 — Counterfactual Visual Explanations

Instanzbasierte Feature-Zell-Vertauschung: diskriminative räumliche Zellen eines
realen Distraktor-Bildes der Zielklasse werden in die Feature-Map des Query
kopiert, bis die Vorhersage kippt.

### 2.1 Was originalgetreu ist

- **Greedy-Exhaustive-Search:** Entspricht `compute_counterfactual` der
  Meta-Referenz mit `lambd=0` und `topk=None` — also der reinen
  Goyal-Baseline ohne den semantischen Auxiliary-Term von Vandenhende et al. In
  jeder Iteration werden alle verbleibenden (Query-Zelle, Distraktor-Zelle)-Paare
  bewertet und der Swap mit maximaler Zielklassen-Wahrscheinlichkeit committet.
- **Auswahlkriterium:** Die Referenz maximiert `log(softmax)[:, distractor_class]`,
  die Projekt-Version `softmax[:, target]`. Da `log` streng monoton ist, ist das
  `argmax` **identisch**.
- **Stopp-Kriterium:** Erste Iteration, in der die argmax-Vorhersage die
  Zielklasse ist ([`run_goyal_cve_pytorch.py:304`](../../scripts/run_goyal_cve_pytorch.py#L304)).
- **Permutations-Constraint:** Jede Query-Zelle wird höchstens einmal editiert,
  jede Distraktor-Zelle höchstens einmal verwendet — identisch zur
  Edit-Filterung `i != query_cell and j != distractor_cell` der Referenz.
- **Feature-Split f / g:** ResNet-18 wird in räumlichen Extraktor (Conv-Stem bis
  `layer4`, `[B,512,7,7]`) und Entscheidungskopf (Global-Average-Pooling + FC)
  zerlegt — exakt die Zerlegung des Papers.
- **Inkrementelles Pooling** ([`run_goyal_cve_pytorch.py:307-316`](../../scripts/run_goyal_cve_pytorch.py#L307-L316)):
  `pooled + (f'(j) − f(i))/N`. Mathematisch **exakt äquivalent** zur Referenz,
  die den Kopf `g` auf jeder voll editierten Feature-Map auswertet, weil
  `g = FC ∘ AveragePool` linear im Zellmittel ist. Nur schneller, nicht anders.

### 2.2 Unterschiede zum Original

- **Distraktor-Auswahl (einzige echte Protokoll-Abweichung):** Das Paper
  definiert die Methode für ein gegebenes Paar `(I, I')` mit `I'` aus der
  Zielklasse. Die Meta-Referenz bestimmt die Distraktor-Klasse über die
  Konfusionsmatrix (meist-verwechselte Klasse) und sampelt daraus bis zu
  `max_num_distractors` **zufällige** Bilder als gemeinsamen Zell-Pool. Die
  Projekt-Version nutzt stattdessen den **nächsten korrekt klassifizierten
  Trainings-Nachbarn** der Manifest-Zielklasse (Cosine-Distanz auf
  L2-normalisierten Pooled-Penultimate-Features), also einen
  Nearest-Unlike-Neighbor-Distraktor. Die Edit-Suche selbst bleibt
  originalgetreu; nur die Herkunft des Distraktors ist projektspezifisch (im
  Docstring und in `metadata.json` ausgewiesen).
- **Framework/Backbone:** ResNet-18 statt der ImageNet-CNNs des Papers; ein
  einzelnes Distraktor-Bild statt eines Multi-Bild-Pools.

### 2.3 Bewusste Design-Entscheidungen

- **`lambd=0`, kein Auxiliary-Modell:** Es wird bewusst die reine
  Goyal-Baseline reproduziert, nicht die ECCV-2022-Erweiterung von Vandenhende
  et al. mit semantischer Konsistenz über ein selbstüberwachtes Hilfsmodell.
- **`max_edits`-Cap (Default = volles 7×7-Gitter = 49):** Sicherheitsnetz gegen
  Endlosschleifen. Bei vollständigem Zellenersatz ist die gepoolte Feature gleich
  der des Distraktors, die Vorhersage kippt also garantiert (das Pooling ist
  permutationsinvariant). Die Referenz kann bei leerem Edit-Set stattdessen
  abstürzen.
- **Feste, manifest-gesteuerte Zielklasse und Cosine-NUN-Retrieval:** dienen der
  fairen, reproduzierbaren Gegenüberstellung mit den anderen drei Methoden auf
  demselben Evaluations-Manifest.
- **Pixel-Komposit zur Visualisierung:** Der entscheidungskippende Edit passiert
  im Feature-Raum; das Komposit-Bild fügt die zu den getauschten 7×7-Zellen
  gehörenden Bildpatches ein. Das ist die Standard-Visualisierung des Papers;
  die Pixel-Änderungsmetriken werden auf diesem Komposit berechnet.

---

## 3. SEDC-T (Search for Explanations by Directed Contrast, targeted)

Best-First-Suche über Superpixel-Segmente: das perturbierte Bild wird schrittweise
um das Segment erweitert, das die Differenz aus Zielklassen- und
Originalklassen-Score am stärksten erhöht, bis die Vorhersage die Zielklasse ist.
Nahezu 1:1-Port von `sedc_t2_fast.py`.

### 3.1 Was originalgetreu ist

- **Best-First-Suche:** Initiales Level mit allen Einzelsegmenten; danach
  wiederholte Expansion des Pending-Kandidaten mit maximalem Expansions-Score,
  bis ein gültiges Level gefunden ist.
- **Expansions-Score:** `p_target − p_originalklasse` auf dem perturbierten
  Kandidaten, wobei die Originalklasse die Vorhersage des **Originalbilds** ist —
  exakt das `p_new_list − results[:, c]` der Referenz.
- **Abbruch-Logik:** Stopp bei gefundenem gültigen CF, bei leerer Expansion
  (keine Kinder mehr) oder bei Timeout — inklusive der Original-Eigenheiten:
  keine Deduplizierung von Segmentmengen, Timeout-Prüfung nur einmal pro
  Expansions-Level, expandierter Knoten wird aus dem Pending-Set entfernt
  ([`run_sedc_t_pytorch.py:398-511`](../../scripts/run_sedc_t_pytorch.py#L398-L511)).
- **Alle vier Replacement-Modi exakt:** kanalweises Mittel; `GaussianBlur
  (31,31), 0`; uniform-random; Navier-Stokes-Inpainting (`cv2.INPAINT_NS`,
  Radius 3) pro Segment.
- **Best-Explanation-Auswahl:** `argmax(P − p)` = höchster Zielklassen-Score-
  Anstieg unter den gültigen Kandidaten, First-on-Tie — exakt Zeile 115 der
  Referenz.
- **Explanation-Bild:** die gewählten Segmente auf schwarzem Hintergrund, wie im
  Original.
- **Segmentierung:** Quickshift mit `kernel_size=4, max_dist=200, ratio=0.2` —
  exakt die Werte aus den Original-Experimentskripten des Repos
  (`experiment_*.py`, `gen_t2_mnv2.py`).
- **Batched Prediction:** Alle Kandidaten eines Levels in einem Forward-Pass
  entsprechen `classifier.predict(cf_candidates)` und sind numerisch äquivalent
  zur sequentiellen Auswertung; die Kandidatenreihenfolge bleibt erhalten, damit
  `max`/`argmax` Ties identisch zum Original bricht.

### 3.2 Unterschiede zum Original

- **Framework/Eingabe:** PyTorch + ResNet-18 statt Keras; Suche im
  `[0,1]`-Pixelraum mit anschließender ImageNet-Normalisierung im Forward-Pass.
- **Verhalten bei Fehlschlag:** Die Referenz gibt `None` zurück. Die
  Projekt-Version behält zusätzlich den besten **ungültigen** Versuch für das
  Reporting (gleiche „No CF found on the requested parameters"-Meldung), zählt
  ihn aber korrekt als ungültig.

### 3.3 Bewusste Design-Entscheidungen

- **Timeout-Default 30 s statt 600 s:** Die Referenz nutzt `max_time=600`. Auf
  den 224×224-Medizinbildern wird jeder gültige CF innerhalb weniger Sekunden
  gefunden; 30 s begrenzt die Wartezeit bei No-CF-Fällen mit großem Sicherheits-
  abstand, ohne einen einzigen CF zu verwerfen. Konfigurierbar; `600` matcht die
  Referenz exakt, `0` deaktiviert den Timeout. Für einen fairen Vergleich über
  Datensätze hinweg identisch zu halten.
- **Lung-Field-ROI-Ablation (`--roi_mode lung_fields`):** Eine **projekt-
  spezifische Ergänzung**, nicht Teil des Original-SEDC-T. Ein grober,
  inhaltsunabhängiger geometrischer Lungenfeld-Prior für frontale Thorax-
  Röntgenbilder, der die auswählbaren Segmente einschränkt. Per Default **aus**
  (`--roi_mode none` = Original-Stil). Klar in `metadata.json`
  (`method_fidelity_note`) und im README als Ablation gekennzeichnet.
- **Konfigurierbare Quickshift-Parameter:** als CLI-Argumente exponiert, aber mit
  den Original-Werten als Default vorbelegt.
- **`blur` als Replacement-Default:** entspricht dem am häufigsten genutzten
  Modus der Original-Experimente; alle anderen Modi sind verfügbar.

---

## 4. DVCE (Diffusion Visual Counterfactual Explanations)

Diffusions-geführte generative Counterfactuals: ein unkonditionaler
OpenAI-256×256-Diffusionsprozess wird durch den Gradienten des medizinischen
Klassifikators (plus optional einem robusten zweiten Klassifikator via
Cone-Projektion) und einen Lp-Distanzterm geführt. Portierung der DVCE-Kernlogik
aus `dff_attack.py` (`DiffusionAttack`).

### 4.1 Was originalgetreu ist

- **`cond_fn_clean`** ([`dvce_core.py:315-396`](../../src/dvce_core.py#L315-L396)):
  rekonstruiert intern `p_mean_variance` (`clip_denoised=False`), wertet
  Klassifikator und Lp-Distanz auf `pred_xstart` aus, füttert **ungeclampte**
  Werte durch `_map_img` (`0.5·(x+1)` ohne Clamp), damit Gradienten auch für
  Pixel außerhalb des Bereichs fließen. Gibt `grad_out = lambda·grad_class −
  lp_value·lp_grad` zurück — exakt die Original-Struktur.
- **Klassifikator-Gradient:** Ziel-log-Softmax gemittelt über die Augmentierungen,
  mit `y.view(-1).repeat(aug_num)`-Indexierung — identisch zum Original.
- **Lp-Distanzterm:** beide Zweige repliziert — geschlossener Lp-Gradient
  (`compute_lp_gradient`) im Standardfall, Autograd auf `compute_lp_dist` bei
  `--denoise_dist_input`. Distanz stets gegen das `init_image`.
- **`enforce_same_norms`:** `_renormalize_gradient` normalisiert Klassifikator-
  und Distanzgradient **getrennt** auf die Norm von `eps = model_output`, exakt
  wie im Original (`condition_mean` reicht `p_mean_var['model_output']` als `eps`
  in die `cond_fn`).
- **Cone-Projektion** ([`dvce_core.py:188-246`](../../src/dvce_core.py#L188-L246)):
  funktionsgleiche Kopie inklusive der Berechnung auf geflachten CPU-Tensoren.
  Die Argument-Reihenfolge entspricht exakt `dff_attack.py`: Der Gradient des
  robusten Helfer-Klassifikators wird auf den Kegel um den Gradienten des
  erklärten Klassifikators projiziert.
- **`ImageAugmentations`:** wird byte-identisch als Original-Klasse per Dateipfad
  geladen (umgeht nur die schweren Package-Importe wie lpips/tensorboard). Bei
  `aug_num=1` ist es die Identität, exakt wie im Original.
- **Klassifikator-Adapter (`MedicalResNetAdapter`):** spiegelt
  `ResizeAndMeanWrapper` — bicubische Resize auf 224 (`interpolation=3`),
  ImageNet-Normalisierung, **kein** Input-Clamping.
- **Sampling:** die vendored `p_sample_loop_progressive` mit `eps`-Übergabe über
  `condition_mean` und Seed-Reseeding; das finale Bild ist `pred_xstart` des
  letzten Schritts, `_map_img` + Clamp auf `(0,1)`.
- **Backbone-Konfiguration & Gradienten-Regel:** `attention_resolutions
  "32,16,8"`, `learn_sigma`, `num_channels=256` usw.; nach dem Laden werden nur
  `qkv`/`norm`/`proj`-Parameter auf `requires_grad=True` gesetzt — exakt die
  `DiffusionAttack`-Reihenfolge (freeze → eval → to(device) → Gradienten
  reaktivieren → optional fp16).
- **Hyperparameter-Defaults** treffen die Original-`default.yml` exakt:
  `timestep_respacing=200`, `skip_timesteps=100`, `lp_custom=1.0`,
  `lp_custom_value=0.15`, `classifier_lambda=0.1`, `enforce_same_norms=true`,
  `gen_type=p_sample`, `seed=1`, `model_output_size=256`.

### 4.2 Unterschiede zum Original

- **Rollen-Vertauschung `classifier` / `second_classifier`:** Im Projekt ist
  `classifier` das erklärte medizinische Modell und `second_classifier` der
  robuste Helfer. Im Original-Vokabular ist die Zuordnung umgekehrt. Die
  **Kegel-Geometrie bleibt identisch** (der robuste Gradient wird auf den Kegel
  um den erklärten Gradienten projiziert); die Vertauschung ist im Docstring
  erklärt und durchgängig konsistent.
- **Weggelassene, im DVCE-Config ohnehin deaktivierte Pfade:** `layer_reg`,
  LPIPS-/L2-Similarity und der Blended-Diffusion-Zweig sind nicht portiert —
  in `default.yml` sind diese Gewichte 0.
- **Framework-Details fürs Reporting:** zusätzliche Metriken (mittlere absolute
  Differenz, geänderte Pixel usw.) und Manifest-gesteuerte Sample-Auswahl, die
  das Original so nicht kennt.

### 4.3 Bewusste Design-Entscheidungen

- **`denoise_dist_input` in den originaltreuen Kommandos aktiviert:** Der
  Argparse-Default des Originals ist `False`, aber **beide** publizierten
  DVCE-Kommandos im Original-README setzen `--denoise_dist_input`. Die
  originaltreuen Projekt-Runs aktivieren es daher ebenfalls (im `debug`-Feld
  vermerkt).
- **Cone-Projektion als originaltreue Hauptvariante für das nicht-robuste
  ResNet-18:** `--deg_cone_projection 30 --aug_num 16` mit einem robusten
  PGD-ResNet-18 als zweitem Klassifikator — spiegelt das zweite Original-README-
  Kommando exakt.
- **MPS-CPU-Roundtrip in den Augmentierungen:** reiner Device-Workaround (MPS
  unterstützt kein nicht-teilbares Adaptive-Pooling). Hält den Autograd-Graphen
  intakt und ändert die Methode nicht; CUDA-Runs bleiben on-device.
- **Per-Sample-Seed `seed + sample_index`:** Reproduzierbarkeit bei
  Einzelbild-Generierung statt eines festen Batch-Seeds. Reines
  Evaluationsdetail.
- **RNG-Detail bei der zweiten Augmentierung:** Für den zweiten Klassifikator
  wird eine frische Augmentierung gezogen; die Reihenfolge der Zufallsziehungen
  kann gegenüber dem Original abweichen. Rein stochastisch, methodisch identisch.
- **Medizinischer Diffusions-Checkpoint optional:** neben dem OpenAI-Checkpoint
  kann ein feingetuntes Diffusionsmodell übergeben werden (`--diffusion_checkpoint_path`).

---

## 5. Zusammenfassung der bewussten Abweichungen

Die folgenden drei Punkte sind die einzigen, die in einer Veröffentlichung
explizit als bewusste Abweichung vom Original ausgewiesen werden sollten (in
Metadata/Docstrings bereits größtenteils dokumentiert):

| # | Methode | Abweichung | Auswirkung |
| --- | --- | --- | --- |
| 1 | Goyal CVE | Nearest-Unlike-Neighbor-Distraktor (Cosine auf Pooled-Features) statt zufälligem Konfusionsmatrix-Sampling | Edit-Suche identisch; nur die Distraktor-Herkunft ist projektspezifisch |
| 2 | SEDC-T | Such-Timeout 30 s statt 600 s | Kein CF geht verloren; nur die Wartezeit bei No-CF-Fällen ist gedeckelt |
| 3 | CFProto | Hyperparameter nach Alibi-MNIST-Beispiel (`c_init=1, c_steps=2`), dimensionsbedingt reskaliertes `gamma`/`theta` | Algorithmus identisch; nur Aufwand/Gewichtung an 224×224 angepasst |

Alle übrigen Unterschiede sind Framework- (TF→PyTorch), Klassifikator-
(ImageNet→Medizin) oder Reporting-bedingt und projektweit gewollt.

**Wichtiger Reporting-Hinweis:** „Model-Validity" bedeutet, dass der
Klassifikator die Zielklasse vorhersagt. Es impliziert **keine** medizinische
Plausibilität oder klinische Kausalität.

---

## 6. Projektspezifische Ergänzungen (kein Teil der Originalmethoden)

- **SEDC-T Lung-Field-ROI:** geometrischer Lungenfeld-Prior als Ablation, per
  Default aus.
- **Fixe Evaluations-Manifeste:** identische Samples und Zielklassen über alle
  vier Methoden hinweg für einen fairen Vergleich.
- **Einheitliche Änderungs-/Validitätsmetriken** über alle Methoden.
- **DVCE-Cone-Projektion mit robustem PGD-ResNet-18** als zweitem Klassifikator
  für den nicht-robusten medizinischen Klassifikator.
