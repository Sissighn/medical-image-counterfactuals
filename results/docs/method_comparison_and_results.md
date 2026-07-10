# Methodenvergleich & Ergebnisse

Diese Datei ist die zentrale, handgepflegte Zusammenfassung des Methodenvergleichs
für das Seminarprojekt: welche vier Counterfactual-Methoden verglichen werden,
ihre quantitativen Ergebnisse, die Interpretation, sowie die Frage, welche
Methoden-Varianten behalten bzw. ersetzt wurden und wie sie zu benennen sind.

Sie führt die früheren Einzeldateien `final_method_summary.md`,
`method_comparison.md` und `method_variant_rationale.md` zusammen.

**Verwandte Dokumente (nicht hier dupliziert):**
- Kanonische, **automatisch generierte** Zahlentabelle:
  [`fixed_evaluation_summary.md`](fixed_evaluation_summary.md) (aus den
  `metadata.json` per `scripts/summarize_counterfactual_evaluation.py`).
- Originaltreue-Audit je Methode:
  [`method_fidelity_comparison.md`](method_fidelity_comparison.md).
- Code-, Funktions- und Parameter-Walkthrough:
  [`notebooklm_code_walkthrough.md`](notebooklm_code_walkthrough.md).
- Methoden-Detaildokus und Run-Kommandos: [`results/final_configs/`](../final_configs/).

**Grundprinzip fürs Reporting:** „Validity" bedeutet ausschließlich, dass der
Klassifikator die Zielklasse vorhersagt. Es impliziert **keine** medizinische
Plausibilität, klinische Kausalität oder dass die hervorgehobene Bildänderung ein
menschlich interpretierbarer Krankheitsmarker ist.

---

## 1. Verglichene Methoden

Das Projekt vergleicht vier Counterfactual-Richtungen für die medizinische
Bildklassifikation:

1. **CFProto (original-style)** — optimierungsbasiert (prototyp-geführt)
2. **Goyal et al. 2019 CVE** — instanzbasiert (Feature-Zell-Tausch)
3. **SEDC-T** — regionenbasiert (Segment-Ersatz)
4. **DVCE (original-style)** — generativ (diffusionsgeführt)

Alle Methoden laufen auf denselben festen Evaluations-Manifesten (BUSI 15,
Pneumonia 20), also identischen Samples und Zielklassen. DVCE ist pro Sample
teurer (Diffusions-Sampling), wird aber ebenfalls auf den vollen Manifesten mit
dem original-code-näheren Kern ausgeführt; die originaltreue Variante für das
nicht-robuste ResNet-18 ist die Cone-Projektion.

## 2. Baseline-Klassifikatoren

| Datensatz | Klassen | Modell | Accuracy | Weighted F1 |
| --- | --- | --- | ---: | ---: |
| BUSI | benign, malignant, normal | ResNet18 pretrained | 0.8390 | 0.8365 |
| Pneumonia | NORMAL, PNEUMONIA | ResNet18 pretrained | 0.8782 | 0.8732 |

---

## 3. Methode 1 — CFProto (original-style prototyp-geführte Optimierung)

Optimiert das Bild im Pixelraum. Folgt Alibis `CounterfactualProto` originalgetreu:

- FISTA-Optimierung mit Shrinkage-Thresholding und Nesterov-Momentum,
- Hinge-Attack-Loss, der die Originalklasse unter die beste andere Klasse drückt,
- summenbasierte Loss `c·L_attack + L2 + beta·L1 + gamma·L_AE + theta·L_proto`,
- binäre Suche über die Attack-Konstante `c` (×10-Eskalation),
- Encoder-Raum-Klassenprototypen aus den eigenen Vorhersagen des Klassifikators
  auf dem Trainingssplit (kNN-Mittel),
- Elastic-Net-Auswahl (`L2 + beta·L1`) des besten Counterfactuals.

**Bewusste Abweichungen:** nur das Framework (PyTorch statt TensorFlow-1.x-Graph)
und die pro Datensatz/Autoencoder rekalibrierten `gamma`/`theta`-Gewichte — da
alle Loss-Terme Summen sind, hängt ihre Rohgröße von der Eingabe- und
Latent-Dimension ab, weshalb die MNIST-Beispielwerte des Originals nicht
übertragbar sind. **Nicht reproduziert:** der TensorFlow-Graph selbst, der
Black-Box-Modus mit numerischen Gradienten, kategoriale Variablen/k-d-Baum-
Prototypen und der TrustScore-Filter (in Alibi per Default ohnehin deaktiviert).
Vollständiger Soll-Ist-Abgleich:
[`cfproto_encoder_method_documentation.md`](../final_configs/cfproto_encoder_method_documentation.md).

| Datensatz | Samples | Validity | Ø CF-Confidence | Ø geänderter Pixelanteil | Ø Laufzeit |
| --- | ---: | ---: | ---: | ---: | ---: |
| BUSI | 15 | 0.87 | 0.6815 | 0.0529 | 46.10 s |
| Pneumonia | 20 | 1.00 | 0.5740 | 0.0180 | 46.34 s |

**Interpretation:** Die zwei BUSI-Fehlschläge (von 15) sind eine erwartete Folge
des Hinge-Attack-Loss: Die Optimierung fand einen sicheren Wechsel *weg* von der
Originalklasse, der aber nicht auf der im Manifest fixierten Zielklasse landete
(BUSI mit `theta=0.5`: 0.87; Pneumonia mit `theta=0.05`: 1.00). Diese
Konfiguration ersetzte die früheren Feature-Map-, Bottleneck-1024- und
ResNet-/Klassenmittel-Prototyp-Experimente, die nicht mehr als eigene
Vergleichszeilen geführt werden.

## 4. Methode 2 — Goyal et al. 2019 Counterfactual Visual Explanations

Instanzbasierter Feature-Raum-Edit nach Goyal et al. (ICML 2019,
arXiv:1904.07451). Das ResNet-18 wird in einen räumlichen Extraktor (`layer4`,
7×7×512 Zellen) und einen Entscheidungskopf (GAP + FC) zerlegt. Ein
Distraktor-Bild der Zielklasse wird als der nächste korrekt klassifizierte
Trainings-Nachbar im gepoolten Feature-Raum abgerufen; danach werden räumliche
Zellen der Query-Feature-Map greedy gegen Distraktor-Zellen getauscht (jede Zelle
höchstens einmal), bis die Vorhersage zur Zielklasse kippt. Referenzimplementierung
ist die Goyal-Baseline im Meta-Repo `facebookresearch/visual-counterfactuals`.

| Datensatz | Samples | Validity | Ø CF-Confidence | Ø Edits (von 49) | Ø geänderter Pixelanteil | Ø Laufzeit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BUSI | 15 | 1.00 | 0.5279 | 14.0 | 0.2596 | 0.25 s |
| Pneumonia | 20 | 1.00 | 0.5231 | 16.15 | 0.3072 | 0.17 s |

**Interpretation:** Validity ist per Konstruktion 1.00 — mit dem vollen
49-Zellen-Budget konvergiert die gepoolte Feature zur der des Distraktors, sodass
die Vorhersage garantiert kippt. Die aussagekräftige Kenngröße ist daher die
**Anzahl editierter Zellen** (Sparsity: Ø 14.0 auf BUSI, 16.15 auf Pneumonia von
49). Die Ø CF-Confidence liegt nahe 0.5, weil die Greedy-Suche beim ersten Flip
stoppt (Sparsity vor Margin). Die Edits sind in einem realen Distraktor-Bild der
Zielklasse verankert und auf ein grobes 7×7-Gitter lokalisiert. Diese Methode
ersetzte die frühere reine Retrieval-NUN-Baseline, die nur den nächsten
Nachbarn abrief, ohne das Query zu editieren, und nicht auf einer publizierten
Originalmethode beruhte. Siehe
[`goyal_cve_method_documentation.md`](../final_configs/goyal_cve_method_documentation.md).

## 5. Methode 3 — SEDC-T-artiger Segment-Ersatz

SEDC-T verändert Bildsegmente und fragt den Klassifikator auf einen
Zielklassen-Flip ab. Der original-style Best-First-Lauf ist die
Methodentreue-Referenz. Behalten werden **zwei Zustände**, weil sie
unterschiedliche Fragen beantworten:

- **original-style Best-First:** näher am referenzierten SEDC-T-Suchmechanismus,
- **Pneumonia Lung-Field-ROI-Ablation:** derselbe Best-First-/Quickshift-/
  Gauß-Blur-Mechanismus, aber die Kandidatensegmente sind auf eine einfache
  geometrische Lungenfeld-Maske beschränkt.

| Variante | Datensatz | Samples | Validity | Ø CF-Confidence | Ø geänderter Pixelanteil | Ø Laufzeit |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Original-style Best-First | BUSI | 15 | 0.80 | 0.6343 | 0.2640 | 6.71 s |
| Original-style Best-First | Pneumonia | 20 | 0.55 | 0.6759 | 0.3270 | 13.92 s |
| Lung-Field-ROI-Ablation | Pneumonia | 20 | 0.50 | 0.7770 | 0.1745 | 15.23 s |

**Interpretation:** SEDC-T liefert lokalisierte Änderungen auf Segmentebene und
ist visuell oft gut diskutierbar. Die Validity ist niedriger, besonders auf
Pneumonia, wo diffuse Modell-Hinweise den Segment-Ersatz erschweren. Der
original-style Best-First-Lauf ist als Methodentreue-Referenz zu verwenden; das
ROI-Ergebnis ist **ausdrücklich als projektspezifische Pneumonia-Ablation** zu
beschreiben — nicht als Teil des Original-SEDC-T und nicht als medizinische
Lungensegmentierung. Siehe
[`sedc_t_method_documentation.md`](../final_configs/sedc_t_method_documentation.md).

## 6. Methode 4 — DVCE diffusionsgeführte Generierung

DVCE deckt die generative Richtung ab und nutzt den original-code-näheren Kern
([`src/dvce_core.py`](../../src/dvce_core.py)):

- `gen_type=p_sample`,
- `timestep_respacing=200`, `skip_timesteps=100`,
- `classifier_lambda=0.1`, `lp_custom=1.0`, `lp_custom_value=0.15`,
- `enforce_same_norms=True`, `clip_denoised=False`,
- Cone-Projektion via `--second_model_path` (PGD-robustes ResNet-18) und
  `--deg_cone_projection 30`, `--aug_num 16`.

**Zwei Achsen:** (a) Guidance-Zustand — **Cone-Projektion** ist die
originaltreue Variante für das nicht-robuste erklärte ResNet-18 (das Original
erklärt nicht-robuste Modelle nur via Cone-Projektion); **ohne Cone-Projektion**
ist nur als ausgewiesene **Ablation** behalten (original ohne Cone ist für robuste
Klassifikatoren definiert). (b) Diffusions-Checkpoint — **OpenAI** 256×256
unkonditional (Original-Backbone), **Pneumonia-** und **BUSI-feingetunte**
medizinische EMA-Checkpoints.

| Variante | Checkpoint | Datensatz | n | Validity | Ø CF-Conf. | Ø abs. Diff | Geänderte Px (>0.05) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Cone (originaltreu) | OpenAI | BUSI | 15 | 0.93 | 0.944 | 0.024 | 0.116 |
| Cone (originaltreu) | OpenAI | Pneumonia | 20 | 0.80 | 0.837 | 0.017 | 0.051 |
| Cone | BUSI feingetunt | BUSI | 15 | 1.00 | 0.998 | 0.028 | 0.156 |
| Cone | Pneumonia feingetunt | Pneumonia | 20 | 1.00 | 0.980 | 0.019 | 0.067 |
| No-Cone (Ablation) | BUSI feingetunt | BUSI | 15 | 1.00 | 0.998 | 0.026 | 0.136 |
| No-Cone (Ablation) | Pneumonia feingetunt | Pneumonia | 20 | 1.00 | 0.995 | 0.017 | 0.052 |
| No-Cone (Ablation) | OpenAI | Pneumonia | 20 | 1.00 | 0.997 | 0.018 | 0.060 |

**Interpretation:** Der Kern entspricht dem originalen `dff_attack.py`:
`p_sample`, Klassifikator- und Distanz-Guidance auf `pred_xstart` (ungeclamptes
`_map_img`), eps-Norm-Rebalancing bei `enforce_same_norms=True` und
Cone-Projektion, die den Gradienten des robusten PGD-Klassifikators auf den Kegel
um den Gradienten des erklärten Klassifikators projiziert. Die feingetunten
Checkpoints erreichen volle Validity (1.00); der generische OpenAI-Checkpoint ist
niedriger (0.93 BUSI, 0.80 Pneumonia), was erwartbar ist, da er auf natürlichen
Bildern statt medizinischen Scans trainiert wurde. **Laufzeithinweis:** Die
OpenAI-Laufzeiten (700–1173 s) spiegeln eine CPU-gebundene Maschine wider und sind
nicht maschinenübergreifend vergleichbar; die feingetunten Läufe (~33–45 s) sind
repräsentativ. Siehe
[`dvce_method_documentation.md`](../final_configs/dvce_method_documentation.md)
und [`dvce_cone_projection.md`](../final_configs/dvce_cone_projection.md).

---

## 7. Zusammenfassende Vergleichstabelle

| Methode | Datensatz | Samples | Validity | Ø CF-Confidence | Ø Änderung | Ø Laufzeit |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| CFProto (original-style) | BUSI | 15 | 0.87 | 0.6815 | 0.0529 Pixelanteil | 46.10 s |
| CFProto (original-style) | Pneumonia | 20 | 1.00 | 0.5740 | 0.0180 Pixelanteil | 46.34 s |
| Goyal et al. 2019 CVE | BUSI | 15 | 1.00 | 0.5279 | 0.2596 Pixelanteil, 14.0 Edits | 0.25 s |
| Goyal et al. 2019 CVE | Pneumonia | 20 | 1.00 | 0.5231 | 0.3072 Pixelanteil, 16.15 Edits | 0.17 s |
| SEDC-T original-style Best-First | BUSI | 15 | 0.80 | 0.6343 | 0.2640 Pixelanteil | 6.71 s |
| SEDC-T original-style Best-First | Pneumonia | 20 | 0.55 | 0.6759 | 0.3270 Pixelanteil | 13.92 s |
| SEDC-T Lung-Field-ROI-Ablation | Pneumonia | 20 | 0.50 | 0.7770 | 0.1745 Pixelanteil | 15.23 s |
| DVCE Cone (OpenAI, originaltreu) | BUSI | 15 | 0.93 | 0.944 | 0.116 Pixelanteil | 1173.4 s |
| DVCE Cone (OpenAI, originaltreu) | Pneumonia | 20 | 0.80 | 0.837 | 0.051 Pixelanteil | 700.2 s |
| DVCE Cone (feingetunt) | BUSI | 15 | 1.00 | 0.998 | 0.156 Pixelanteil | 44.6 s |
| DVCE Cone (feingetunt) | Pneumonia | 20 | 1.00 | 0.980 | 0.067 Pixelanteil | 44.9 s |

Die vollständige, automatisch generierte Tabelle (inkl. aller No-Cone-Zeilen und
Metadata-Pfade) steht in
[`fixed_evaluation_summary.md`](fixed_evaluation_summary.md).

---

## 8. Behaltene vs. ersetzte Varianten (Rationale)

| Methodenfamilie | Behaltene Rolle | Anmerkungen |
| --- | --- | --- |
| CFProto (original-style) | Finale prototyp-geführte Methode | FISTA + Shrinkage-Thresholding, Hinge-Attack-Loss, Encoder-Raum-Prototypen, binäre c-Suche, Elastic-Net-Auswahl. Ersetzt Feature-Map-, Bottleneck-1024- und Klassenmittel-Experimente |
| Goyal et al. 2019 CVE | Instanzbasierter Feature-Raum-Edit | Greedy-Zell-Swaps von einem Nearest-Unlike-Distraktor; ersetzt die frühere reine Retrieval-NUN-Baseline |
| SEDC-T | Regionenbasierte/lokalisierte CFs | Original-style Best-First + Pneumonia Lung-Field-ROI-Ablation |
| DVCE | Generative diffusionsgeführte CFs | Cone-Projektion (originaltreu für nicht-robustes ResNet-18) + No-Cone-Ablation; Checkpoints OpenAI, Pneumonia-medizinisch, BUSI-medizinisch |

## 9. Empfohlene Benennung (für Arbeit/Vortrag)

- CFProto: „CFProto original-style prototyp-geführte Counterfactuals".
- Goyal: „Goyal et al. 2019 Counterfactual Visual Explanations".
- SEDC-T: original-style Best-First als Treue-Referenz; ROI stets als
  „projektspezifische Pneumonia-Ablation" kennzeichnen.
- DVCE: Cone-Projektion als originaltreue Hauptvariante; „ohne Cone-Projektion"
  stets als **Ablation** kennzeichnen; Checkpoint-Achse getrennt nennen.

---

## 10. Haupterkenntnis

Die Methoden legen unterschiedliche Trade-offs offen:

- **CFProto** ist kompakt und meist modell-valide (kleine, plausibilitäts-
  regularisierte Änderung).
- **Goyal et al. CVE** liefert dünne, lokalisierte Edits, verankert in realen
  Bildern der Zielklasse (Validity per Konstruktion garantiert, Confidence nahe
  der Entscheidungsgrenze).
- **SEDC-T** ist lokalisierter, aber weniger konsistent valide.
- **DVCE** ist generativ und erreicht mit den medizinisch feingetunten Checkpoints
  volle Validity, mit dem generischen OpenAI-Checkpoint niedriger (0.80–0.93).

Modell-Validity darf nicht mit medizinischer Plausibilität gleichgesetzt werden —
diese ist stets getrennt zu diskutieren.
