# Code-, Funktions- und Parameter-Walkthrough (für NotebookLM & Prüfungsvorbereitung)

Dieses Dokument erklärt den **tatsächlichen Code** der vier Counterfactual-
Methoden in Prosa: welche Funktion was tut, in welcher Reihenfolge der Ablauf
läuft, und was **jeder** Parameter bewirkt (Default, Wertebereich, Effekt). Es
ergänzt die konzeptionellen Methodendokus in
[`results/final_configs/`](../final_configs/) und den Originaltreue-Vergleich
[`method_fidelity_comparison.md`](method_fidelity_comparison.md), indem es die
Brücke von der Prosa-Beschreibung zum konkreten Python-Code schlägt.

Ziel: Nach dem Durcharbeiten (bzw. nach NotebookLM-Fragen darauf) sollst du jede
Detailfrage der Art „Was macht diese Funktion / dieser Parameter genau und
warum?" beantworten können.

**Terminologie-Hinweis:** „Validity" heißt immer nur, dass der Klassifikator die
Zielklasse vorhersagt — **nicht** medizinische Plausibilität.

---

## 0. Gemeinsame Infrastruktur (von allen vier Methoden genutzt)

Bevor die Methoden erklärt werden, hier die geteilten Bausteine. Sie kommen in
allen vier Skripten vor.

### 0.1 Datensatz-Laden — [`src/data_utils.py`](../../src/data_utils.py)

- **`create_dataloaders(dataset_path, batch_size, use_augmentation=False)`**
  baut Train-/Test-DataLoader und -Datasets über `torchvision.ImageFolder`. Die
  Bilder werden auf `IMAGE_SIZE` (224×224) skaliert und mit ImageNet-Mittelwert/
  -Standardabweichung normalisiert. Für die Counterfactual-Läufe ist
  `use_augmentation=False`, damit die Eingabe deterministisch ist.
- Rückgabe ist ein Dict mit `train_loader`, `test_loader`, `train_dataset`,
  `test_dataset`.

### 0.2 Normalisierung (in jedem Methodenskript wiederholt)

Alle Skripte definieren dieselben zwei ImageNet-Konstanten und Hilfsfunktionen:

```
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
```

- **`normalize(images)`** = `(images − mean) / std`: bringt `[0,1]`-Pixel in den
  normalisierten Raum, den das ResNet erwartet.
- **`denormalize(images)`** = `(images · std + mean).clamp(0,1)`: kehrt das um
  und liefert anzeigbare/vergleichbare `[0,1]`-Pixel.

**Warum das wichtig ist:** Die Counterfactual-Suche und alle Distanzmetriken
laufen im `[0,1]`-Pixelraum; die ImageNet-Normalisierung ist nur ein interner
Schritt direkt vor dem Klassifikator. Das entspricht bei CFProto exakt Alibis
`feature_range=(0,1)`.

### 0.3 Feste Evaluations-Manifeste — [`src/evaluation_manifest.py`](../../src/evaluation_manifest.py)

Damit alle vier Methoden **dieselben** Testbilder und **dieselben** Zielklassen
sehen (fairer Vergleich):

- **`load_manifest_records(manifest_path, max_records)`** lädt die JSON-Liste der
  fixierten Samples.
- **`load_image_from_manifest_record(test_dataset, record)`** holt das konkrete
  Bild + Label + Pfad.
- **`manifest_record_metadata(record)`** extrahiert die Metadaten (z. B.
  `manifest_sample_index`, `dataset_index`, `source_image_path`).
- Jedes Skript prüft: Die **aktuelle** Modellvorhersage muss der im Manifest
  gespeicherten `original_prediction_index` entsprechen — sonst Abbruch mit
  Fehler. Das garantiert, dass die Ausgangslage reproduzierbar ist.

### 0.4 Gerätewahl — `get_device()` aus [`src/train_model.py`](../../src/train_model.py)

Wählt automatisch MPS (Apple), CUDA oder CPU. (Das DVCE-Skript hat eine eigene
`resolve_device`-Variante mit `--device auto`.)

### 0.5 Autoencoder — [`src/autoencoder.py`](../../src/autoencoder.py) (nur CFProto)

Faltungs-Autoencoder in zwei Architekturen: `ConvAutoencoder` (Feature-Map-
Encoder) und `ConvAutoencoderBottleneck` (Flaschenhals mit `latent_dim`). Für die
finale CFProto-Konfiguration wird die Bottleneck-256-Variante genutzt. Der
Encoder liefert den Raum, in dem Klassen-Prototypen und der Prototyp-Loss leben.

---

## 1. CFProto — [`scripts/run_cfproto_pytorch.py`](../../scripts/run_cfproto_pytorch.py)

**Kernidee:** Optimiere das Eingabebild im Pixelraum so, dass (a) der
Klassifikator die Originalklasse verlässt, (b) die Änderung klein bleibt
(Elastic-Net L1+L2), (c) das Bild vom Autoencoder gut rekonstruierbar bleibt
(Plausibilität), und (d) das Bild im Encoder-Raum nahe an den Prototyp einer
Zielklasse gezogen wird. Optimierer ist FISTA. Portierung von Alibis
`CounterfactualProto`.

### 1.1 Wichtige Funktionen und Kontrollfluss

1. **`load_checkpoint_model`** lädt das ResNet-18 und wickelt es in
   **`ResNetWithFeatures`**, das sowohl Logits als auch den vorletzten
   Feature-Vektor zurückgibt.
2. **`load_autoencoder_model`** lädt den passenden Autoencoder (Architektur aus
   dem Checkpoint) und friert seine Parameter ein.
3. **`predict_proba(model, pixels)`** = `softmax(model(normalize(pixels)))`.
   Übernimmt die Rolle von Alibis `predict`; die Normalisierung ist Teil des
   gewickelten Prädiktors.
4. **`fit_class_encodings`** (= Alibis `fit`): Kodiert alle Trainingsbilder,
   klassifiziert sie und gruppiert die Encodings **nach Klassifikator-Vorhersage**
   (nicht nach wahrem Label). Liefert pro Klasse die Encodings (`class_enc`) und
   das Klassenmittel (`class_proto`).
5. **`select_class_prototype`** (= Alibis Prototyp-Wahl in `attack`): Für die
   Kandidaten-Zielklassen wird die Distanz `ENC(x)`→Prototyp berechnet und die
   nächste Klasse gewählt. Bei `k=None` ist der Prototyp das Klassenmittel; bei
   gesetztem `k` das Mittel der k nächsten Encodings, wobei `k_type='mean'` die
   mittlere und `k_type='point'` die k-te Distanz als Auswahlkriterium nimmt.
6. **`cfproto_attack`** — die eigentliche FISTA-Schleife (Kern der Methode):
   - Äußere Schleife über `c_steps` (binäre Suche über die Attack-Konstante `c`).
   - Pro c-Step werden `adv` und `adv_s` auf das Original zurückgesetzt.
   - Innere Schleife über `max_iterations`:
     - **Lernrate** = `learning_rate_init · (1 − i/max_iterations)^0.5`
       (polynomialer Decay, Power 0.5).
     - Gradient der **optimierten Loss** `c·L_attack + L2 + gamma·L_AE +
       theta·L_proto` bzgl. `adv_s`, geclippt auf ±1000.
     - Gradientenschritt, dann **`shrinkage_thresholding`** (bringt den
       `beta·L1`-Term ein und projiziert auf `[0,1]`), dann Nesterov-Momentum
       `zt = (i+1)/(i+4)`.
     - Kandidat `adv` auswerten: `compare()` (kappa-justierter argmax ≠
       Originalklasse) **und** Zielklassen-Mitgliedschaft; wenn gültig und
       Elastic-Net-Distanz kleiner → als bester CF merken.
   - Nach jedem c-Step: **`c` anpassen** — gültige Lösung gefunden → `c`
     verkleinern (mehr Gewicht auf Nähe); keine gefunden → `c` vergrößern
     (×10 bzw. Bisektion).
7. **`compute_change_metrics` / `compute_elastic_net_distance` /
   `compute_loss_terms`** berechnen die Report-Metriken.
8. **`main`** orchestriert: Modell + Autoencoder laden → `fit_class_encodings`
   auf dem Trainingssplit → Samples (Manifest oder korrekt klassifizierte
   Testbilder) → pro Sample Prototyp wählen, `cfproto_attack` laufen lassen,
   Visualisierung + `metadata.json` schreiben.

### 1.2 Hilfsfunktionen im Detail

- **`attack_loss_terms`**: Hinge-Loss `max(0, p_orig − max_{i≠orig} p_i + kappa)`
  mit dem `−10000`-Maskierungstrick.
- **`shrinkage_thresholding`**: drei Fälle — wenn `delta > beta` oberer Zweig,
  wenn `|delta| ≤ beta` zurück auf das Original (das erzeugt die Sparsity), sonst
  unterer Zweig; jeweils auf die Feature-Range projiziert.
- **`compare`**: addiert `kappa` auf die Originalklassen-Wahrscheinlichkeit und
  prüft, ob der argmax dann eine andere Klasse ist.

### 1.3 Parameter

| Parameter | Default | Wirkung |
| --- | --- | --- |
| `--model_path` | (Pflicht) | ResNet-18-Checkpoint des erklärten Klassifikators |
| `--autoencoder_path` | (Pflicht) | Autoencoder-Checkpoint (Encoder- und AE-Rolle) |
| `--dataset_path` | (Pflicht) | Bilddatensatz (ImageFolder) |
| `--output_dir` | (Pflicht) | Ausgabeordner für Bilder + `metadata.json` |
| `--manifest_path` | None | Festes Evaluations-Manifest; wenn gesetzt, Samples + Ziele daraus |
| `--manifest_max_samples` | None | Obergrenze im Manifest-Modus |
| `--max_samples` | 3 | Anzahl Samples ohne Manifest |
| `--max_iterations` | 1000 | FISTA-Iterationen pro c-Step (Alibi-Default) |
| `--learning_rate_init` | 0.01 | Start-Lernrate für den polynomialen Decay (Alibi 1e-2) |
| `--kappa` | 0.0 | Confidence-Margin des Hinge-Loss (Alibi 0) |
| `--beta` | 0.1 | L1-Gewicht, via Shrinkage-Thresholding (Alibi 0.1) |
| `--gamma` | 1.0 | Gewicht des AE-Rekonstruktions-Loss; auf 224×224 reskaliert |
| `--theta` | 0.5 | Gewicht des Prototyp-Loss; auf die Encoder-Dimension reskaliert |
| `--c_init` | 1.0 | Start-Attack-Konstante (Alibi-MNIST-Beispiel 1) |
| `--c_steps` | 2 | Binäre-Suche-Schritte für `c` (Alibi-MNIST-Beispiel 2) |
| `--prototype_k` | None | Anzahl nächster Nachbarn für den Prototyp; None = ganze Klasse |
| `--k_type` | mean | `mean` = mittlere Distanz der k Nachbarn, `point` = k-te Distanz |
| `--target_strategy` | all | `all` = alle Nicht-Original-Klassen als Kandidaten; `second_best` = nur zweitwahrscheinlichste. Im Manifest-Modus ignoriert (dort zählt die Manifest-Zielklasse) |
| `--batch_size` | 16 | Batch-Größe beim Prototyp-Fitting |
| `--verbose` / `--print_every` | aus / 100 | Zwischen-Losses ausgeben |

**Feste Konstanten im Code:** `FEATURE_RANGE=(0.0, 1.0)`,
`GRADIENT_CLIP=(−1000, 1000)` — beide entsprechen Alibi.

### 1.4 Typische Prüfungsfragen

- *Warum ist der L1-Term nicht im Gradienten?* Weil FISTA ihn über das
  Shrinkage-Thresholding einbringt — das erzeugt exakt die gewünschte Sparsity
  und ist Standard bei Elastic-Net-Optimierung.
- *Warum werden Prototypen aus Vorhersagen statt Labels gebildet?* Weil man den
  Counterfactual gegen das **Verhalten des Modells** erklärt, nicht gegen die
  Ground Truth — so macht es auch Alibi.
- *Was bedeutet die c-Suche?* `c` gewichtet den Attack- gegen die
  Regularisierungsterme; die binäre Suche findet den kleinsten `c`, der noch
  einen gültigen CF liefert (→ minimale Änderung).

---

## 2. Goyal et al. CVE — [`scripts/run_goyal_cve_pytorch.py`](../../scripts/run_goyal_cve_pytorch.py)

**Kernidee:** Tausche im Feature-Raum des ResNet einzelne räumliche 7×7-Zellen
des Query-Bildes gegen Zellen eines realen Distraktor-Bildes der Zielklasse aus,
bis die Vorhersage kippt. Kein Pixel-Optimieren, kein Verwischen — echte
Bildinhalte aus einem anderen Bild.

### 2.1 Wichtige Funktionen und Kontrollfluss

1. **`ResNetSpatialSplit`** zerlegt das ResNet-18 in:
   - `spatial_extractor` = alle Layer bis `layer4` → `[B, 512, 7, 7]` (das ist
     `f`),
   - `pool` + `classifier` = Global-Average-Pooling + FC (das ist `g`).
2. **`build_retrieval_database`** kodiert alle Trainingsbilder, behält nur die
   **korrekt klassifizierten**, und speichert pro Klasse deren L2-normalisierte
   Pooled-Features (Distraktor-Kandidaten).
3. **`retrieve_nearest_distractor`** wählt für ein Query den **nächsten** Kandidaten
   der Zielklasse (Cosine-Distanz = `1 − Skalarprodukt` auf den normalisierten
   Features) → Nearest-Unlike-Neighbor.
4. **`greedy_exhaustive_search`** — der Kern:
   - Startet mit der Query-Feature-Map, berechnet die aktuelle gepoolte Feature.
   - Solange die Vorhersage ≠ Zielklasse und `< max_edits`:
     - Bewerte **alle** verbleibenden (Query-Zelle i, Distraktor-Zelle j)-Paare
       über das **inkrementelle Pooling** `pooled + (f'(j) − f(i))/N` (exakt und
       schnell, weil `g` linear im Zellmittel ist).
     - Wähle das Paar mit maximaler Zielklassen-Softmax-Wahrscheinlichkeit.
     - Committe den Swap, markiere Query-Zelle i und Distraktor-Zelle j als
       verbraucht (Permutations-Constraint).
   - Stopp, sobald der argmax die Zielklasse ist.
5. **`build_composite_image`** fügt zur Visualisierung die zu den getauschten
   Zellen gehörenden Bildpatches aus dem Distraktor ins Query ein (Pixel-Komposit).
6. **`main`** orchestriert: Modell → Distraktor-DB → Manifest-Samples → pro
   Sample Distraktor holen, Suche laufen lassen, Komposit + Metriken +
   `metadata.json`.

### 2.2 Parameter

| Parameter | Default | Wirkung |
| --- | --- | --- |
| `--model_path` | (Pflicht) | ResNet-18-Checkpoint |
| `--dataset_path` | (Pflicht) | Bilddatensatz |
| `--manifest_path` | (Pflicht) | Festes Evaluations-Manifest (hier immer Pflicht) |
| `--output_dir` | (Pflicht) | Ausgabeordner |
| `--manifest_max_samples` | None | Obergrenze der Manifest-Records |
| `--batch_size` | 32 | Batch-Größe beim Aufbau der Distraktor-DB |
| `--change_threshold` | 0.03 | Schwelle für „geänderter Pixelanteil" in den Metriken |
| `--max_edits` | None → 49 | Max. Anzahl Zell-Swaps; None = volles 7×7-Gitter. Bei vollem Ersatz kippt die Vorhersage garantiert |

### 2.3 Typische Prüfungsfragen

- *Warum ist Validity hier per Konstruktion ~100 %?* Weil im Extremfall alle 49
  Zellen ersetzt werden und die gepoolte Feature dann der des Distraktors
  entspricht — die Vorhersage muss kippen.
- *Warum inkrementelles Pooling statt Full-Forward?* Rein Effizienz;
  mathematisch identisch, da `g = FC ∘ AvgPool` linear im Zellmittel ist.
- *Wodurch unterscheidet sich das vom Original-Referenzcode?* Nur durch die
  Distraktor-Herkunft (nächster Trainings-Nachbar statt zufälliges
  Konfusionsmatrix-Sampling); die Suche selbst ist identisch, `lambd=0` (kein
  Auxiliary-Modell → reine Goyal-Baseline).

---

## 3. SEDC-T — [`scripts/run_sedc_t_pytorch.py`](../../scripts/run_sedc_t_pytorch.py)

**Kernidee:** Zerlege das Bild in Superpixel-Segmente (Quickshift). Ersetze
schrittweise diejenigen Segmente durch eine „neutrale" Version (verwischt,
Mittelwert, …), die die Zielklasse am stärksten befördern, bis die Vorhersage die
Zielklasse ist. Best-First-Suche. Port von `sedc_t2_fast.py`.

### 3.1 Wichtige Funktionen und Kontrollfluss

1. **`create_segments`** läuft Quickshift (`kernel_size, max_dist, ratio`) und
   liefert die Segmentkarte.
2. **`create_replacement_image`** baut das „perturbierte" Bild je nach Modus:
   `mean` (kanalweiser Mittelwert), `blur` (`GaussianBlur (31,31)`), `random`,
   `inpaint` (Navier-Stokes pro Segment).
3. **`get_allowed_segments`** filtert die auswählbaren Segmente über die ROI
   (bei `--roi_mode none` sind alle erlaubt; `lung_fields` schränkt auf den
   Lungenfeld-Prior ein).
4. **`replace_segments`** setzt ausgewählte Segmente im Bild auf die
   perturbierten Pixel.
5. **`evaluate_segment_sets_batch`** bewertet alle Kandidaten-Segmentmengen eines
   Suchlevels in einem Forward-Pass und berechnet pro Kandidat den
   **Expansions-Score** `p_target − p_originalklasse`.
6. **`generate_sedc_t_original_best_first_counterfactual`** — der Kern:
   - Initial: alle erlaubten Einzelsegmente bewerten. Wird die Zielklasse schon
     erreicht → gültige Kandidaten sammeln.
   - Sonst: wiederhole — nimm den Pending-Kandidaten mit **höchstem**
     Expansions-Score, erweitere ihn um jedes noch nicht enthaltene Segment,
     bewerte die Kinder. Stopp bei gültigem CF, leerer Expansion oder Timeout.
   - Bestauswahl unter den gültigen: `argmax(target_score_increase)` (höchster
     Zielklassen-Score-Anstieg, First-on-Tie).
7. **`main`** orchestriert: Modell → Samples → pro Sample Segmente +
   Replacement-Bild + erlaubte Segmente → Suche pro Zielklasse-Kandidat →
   Visualisierung (inkl. Explanation-Bild = gewählte Segmente auf Schwarz) +
   `metadata.json`.

### 3.2 Parameter

| Parameter | Default | Wirkung |
| --- | --- | --- |
| `--model_path` | (Pflicht) | ResNet-18-Checkpoint |
| `--dataset_path` | (Pflicht) | Bilddatensatz |
| `--output_dir` | (Pflicht) | Ausgabeordner |
| `--manifest_path` | None | Festes Evaluations-Manifest |
| `--manifest_max_samples` | None | Obergrenze im Manifest-Modus |
| `--max_samples` | 3 | Anzahl Samples ohne Manifest |
| `--target_strategy` | all | Zielklassen-Kandidaten ohne Manifest |
| `--quickshift_kernel_size` | 4 | Quickshift-Kernelgröße (Original-Wert) |
| `--quickshift_max_dist` | 200.0 | Quickshift-Maximaldistanz (Original-Wert) |
| `--quickshift_ratio` | 0.2 | Quickshift-Farb/Raum-Gewichtung (Original-Wert) |
| `--search_timeout_seconds` | 30.0 | Timeout pro Ziel, einmal pro Level geprüft. Original nutzt 600; 0 = aus |
| `--replacement_mode` | blur | `mean` / `blur` / `random` / `inpaint` |
| `--blur_kernel` | 31 | Gaußkernel-Größe (muss ungerade sein) |
| `--roi_mode` | none | `none` = Original-Stil; `lung_fields` = Pneumonie-ROI-Ablation |
| `--roi_min_overlap` | 0.50 | Mindest-Überlappung eines Segments mit der ROI |
| `--batch_size` | 16 | Batch-Größe |

### 3.3 Typische Prüfungsfragen

- *Was ist der Unterschied zu SEDC (ohne T)?* Das „T" steht für targeted: die
  Suche optimiert gezielt die **Ziel**klassen-Wahrscheinlichkeit
  (`p_target − p_orig`), nicht nur die Reduktion der Originalklasse.
- *Warum Timeout 30 statt 600 s?* Auf den 224×224-Bildern wird jeder gültige CF
  in Sekunden gefunden; 30 s deckelt nur die No-CF-Fälle, ohne einen CF zu
  verlieren (konfigurierbar).
- *Ist die Lung-Field-ROI Teil von SEDC-T?* Nein — projektspezifische Ablation,
  per Default aus, klar so gekennzeichnet.

---

## 4. DVCE — [`src/dvce_core.py`](../../src/dvce_core.py) + [`scripts/run_dvce_pytorch.py`](../../scripts/run_dvce_pytorch.py)

**Kernidee:** Ein vortrainierter unkonditionaler Diffusionsprozess (OpenAI 256×256
oder medizinisch feingetunt) erzeugt ein neues, realistisches Bild. Der Prozess
wird bei jedem Denoising-Schritt durch den **Gradienten des medizinischen
Klassifikators** in Richtung Zielklasse gelenkt, plus einen **Lp-Distanzterm**,
der nahe am Originalbild hält, optional plus **Cone-Projektion** mit einem
robusten zweiten Klassifikator. Portierung der DVCE-`DiffusionAttack`.

### 4.1 Struktur: Core vs. Runner

- **`src/dvce_core.py`** enthält die eigentliche Methode (Backbone laden,
  Guidance-Funktion, Sampling-Schleife).
- **`scripts/run_dvce_pytorch.py`** ist der Runner: lädt die medizinischen
  Klassifikatoren, wickelt sie in Adapter, wählt Samples (Manifest oder korrekt
  klassifizierte Testbilder), ruft den Core auf, berechnet Metriken, schreibt
  `metadata.json`. Er kennt einen Einzelbild- und einen Multi-Sample-Modus.

### 4.2 Wichtige Funktionen im Core

1. **`load_dvce_diffusion_backbone`** baut das Diffusionsmodell nach der
   Original-Config (`build_dvce_model_config`: `attention_resolutions "32,16,8"`,
   `learn_sigma`, `num_channels=256`, …), lädt den Checkpoint und setzt nur
   `qkv`/`norm`/`proj`-Parameter auf `requires_grad=True` — exakt die
   `DiffusionAttack`-Reihenfolge.
2. **`make_original_style_cond_fn`** erzeugt die Guidance-Funktion
   `cond_fn_clean(x, t, y, eps)`:
   - Rekonstruiert intern `p_mean_variance` und nimmt `x_in = pred_xstart`.
   - **Klassifikator-Gradient**: `map_minus1_1_to_0_1(x_in)` (= `0.5·(x+1)`, ohne
     Clamp) → Klassifikator → Ziel-log-Softmax → Gradient bzgl. `x`.
   - Bei zweitem Klassifikator + `deg_cone_projection > 0`: dessen Gradient wird
     per **`cone_projection`** auf den Kegel um den ersten Gradienten projiziert.
   - **Lp-Distanzterm**: entweder geschlossener Lp-Gradient
     (`compute_lp_gradient`) oder Autograd auf `compute_lp_dist` gegen das
     `init_image` (bei `denoise_dist_input`).
   - Bei `enforce_same_norms`: beide Gradienten getrennt via
     `renormalize_gradient` auf die Norm von `eps = model_output` skaliert.
   - Rückgabe: `grad_out = classifier_lambda·grad_class − lp_custom_value·lp_grad`.
3. **`cone_projection`** — funktionsgleiche Kopie des Originals: berechnet den
   Winkel zwischen den Gradienten und projiziert (nur wenn Winkel > `deg`) auf
   den Kegel.
4. **`generate_dvce_counterfactual`** — Orchestrierung: Backbone laden, Bild auf
   256 skalieren und nach `[−1,1]` bringen (`init_image`), Guidance bauen,
   `p_sample_loop_progressive` (oder `ddim`) mit `skip_timesteps` laufen lassen,
   finales `pred_xstart` per `_map_img` + Clamp zurückgeben.

### 4.3 Wichtige Funktionen im Runner

- **`MedicalResNetAdapter`** spiegelt `ResizeAndMeanWrapper`: bicubische Resize
  auf 224, ImageNet-Normalisierung, **kein** Clamping (damit Guidance-Gradienten
  auch außerhalb `[0,1]` fließen).
- **`validate_second_classifier_checkpoint`** stellt sicher, dass der robuste
  zweite Klassifikator dieselben Klassen hat.
- **`run_generation_for_sample`** führt eine Generierung durch und sammelt alle
  Metriken/Visualisierungen; **`summarize_generation_records`** aggregiert.

### 4.4 Parameter

| Parameter | Default | Wirkung |
| --- | --- | --- |
| `--model_path` | (Pflicht) | Erklärter medizinischer Klassifikator |
| `--second_model_path` / `--second_classifier_path` | None | Robuster zweiter Klassifikator für Cone-Projektion |
| `--dataset_path` | (Pflicht) | Bilddatensatz |
| `--output_dir` | (Pflicht) | Ausgabeordner |
| `--dvce_repo` | external/DVCEs | Pfad zum vendored DVCE-Repo |
| `--diffusion_checkpoint_path` | None → OpenAI | Diffusions-Checkpoint (OpenAI oder medizinisch feingetunt) |
| `--run_generation` | aus | Ohne dieses Flag nur Vorab-Checks, keine echte Generierung |
| `--device` | auto | `auto` / `mps` / `cuda` / `cpu` |
| `--timestep_respacing` | 200 | Anzahl Diffusionsschritte (Original-Default) |
| `--skip_timesteps` | 100 | Wie viele Schritte übersprungen werden (Start näher am Bild) |
| `--model_output_size` | 256 | Auflösung des Diffusionsmodells (256/512) |
| `--classifier_lambda` | 0.1 | Gewicht des Klassifikator-Gradienten (Original 0.1) |
| `--lp_custom` | 1.0 | Ordnung `p` der Lp-Distanz (Original 1.0) |
| `--lp_custom_value` | 0.15 | Gewicht des Distanzterms (Original 0.15) |
| `--enforce_same_norms` | True | Gradienten auf `eps`-Norm normalisieren (Original an) |
| `--denoise_dist_input` | aus | Distanz per Autograd durch den Denoiser; in beiden Original-README-Kommandos gesetzt |
| `--deg_cone_projection` | 0.0 | Kegelöffnung in Grad; > 0 aktiviert Cone-Projektion (braucht zweiten Klassifikator) |
| `--aug_num` | 1 | Anzahl Augmentierungen; 1 = Identität. Cone-Variante nutzt 16 |
| `--clip_denoised` | False | `pred_xstart` während des Samplings clampen (Original aus) |
| `--use_ddim` | aus | DDIM- statt p_sample-Sampling |
| `--diffusion_fp16` | aus | Diffusionsmodell in fp16 |
| `--seed` | 1 | Zufallsseed (Original 1); pro Sample `seed + sample_index` |
| `--num_generation_samples` | 1 | Anzahl Samples im Nicht-Manifest-Modus |
| `--manifest_path` / `--manifest_max_samples` | None | Festes Evaluations-Manifest |

### 4.5 Typische Prüfungsfragen

- *Was macht Cone-Projektion und warum?* Sie kombiniert den (glatten) Gradienten
  eines **robusten** Klassifikators mit dem des erklärten (nicht-robusten)
  Modells: Der robuste Gradient wird auf einen Kegel um den erklärten Gradienten
  projiziert. So bleibt die Guidance in Richtung des erklärten Modells, nutzt
  aber die realistischeren Gradienten des robusten Modells → plausiblere Bilder.
- *Warum `skip_timesteps`?* Man startet nicht bei reinem Rauschen, sondern
  überspringt die ersten Schritte und beginnt näher am Originalbild — hält den
  CF nah am Original und spart Rechenzeit.
- *Warum ist die OpenAI-Checkpoint-Laufzeit so hoch (700–1173 s)?* Reiner
  CPU-Effekt der Testmaschine; die feingetunten Checkpoints laufen in ~33–45 s
  und sind die repräsentativen Zahlen.
- *Welche DVCE-Variante ist die originaltreue Hauptvariante?* Cone-Projektion mit
  robustem PGD-ResNet-18 als zweitem Klassifikator; die „ohne Cone"-Läufe sind
  ausgewiesene Ablationen.

---

## 5. Methoden auf einen Blick (Vergleichstabelle)

| Aspekt | CFProto | Goyal CVE | SEDC-T | DVCE |
| --- | --- | --- | --- | --- |
| Familie | Pixel-Optimierung | Feature-Zell-Tausch | Segment-Ersatz | Generativ (Diffusion) |
| Wo verändert wird | Pixelraum | 7×7-Feature-Map | Superpixel-Segmente | Ganzes Bild neu erzeugt |
| Braucht Zusatzmodell | Autoencoder | Distraktor-Bilder | — | Diffusionsmodell (+ robuster 2. Klassifikator) |
| Optimierer/Suche | FISTA | Greedy-Exhaustive | Best-First | Guided Diffusion |
| Validity typisch | hoch, nicht garantiert | ~100 % (per Konstruktion) | mittel–hoch | hoch |
| Änderungsgröße | klein (sparse) | mittel (lokal) | mittel–groß | klein–mittel, aber global |
| Laufzeit/Sample | ~46 s | ~0,2 s | ~7–14 s | ~33–45 s (feingetunt) |
| Plausibilität | AE-reguliert | reale Distraktor-Inhalte | roh (verwischt) | am realistischsten |

(Die quantitativen Zahlen stammen aus
[`fixed_evaluation_summary.md`](fixed_evaluation_summary.md); dort stehen alle
Datensatz-genauen Werte.)

---

## 6. Was du für die Präsentation parat haben solltest

1. **Ein Satz pro Methode** (siehe „Kernidee" oben) — als Einstieg.
2. **Der Unterschied der vier Ansätze**: optimieren vs. tauschen vs. entfernen
   vs. generieren. Das ist die zentrale Erzähllinie.
3. **Pro Methode eine bewusste Abweichung vom Original** (siehe
   [`method_fidelity_comparison.md`](method_fidelity_comparison.md), Abschnitt 5):
   Goyal-Distraktor, SEDC-T-Timeout, CFProto-Hyperparameter.
4. **Der Validity-Vorbehalt**: Modell-Validity ≠ medizinische Plausibilität — bei
   jeder Ergebnisdiskussion nennen.
5. **Warum vier Methoden?** Sie decken das Spektrum von minimal-invasiv (CFProto)
   bis maximal-realistisch (DVCE) ab und beleuchten den Trade-off zwischen
   kleiner Änderung, Plausibilität und Laufzeit.
