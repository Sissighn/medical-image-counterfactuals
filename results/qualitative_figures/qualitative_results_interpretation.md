# Qualitative Ergebnisinterpretation

Stand: 29.07.2026

## Zweck und Datenbasis

Diese Datei dokumentiert die Interpretation der 14 finalen qualitativen Ergebnisabbildungen. Sie ist auf den aktuellen Ergebnisdateien des Projekts und dem aktuellen Stand von Kapitel 5 der Seminararbeit aufgebaut. Sie ersetzt die frühere, veraltete Interpretationsdatei.

Maßgebliche Grundlagen:

- Seminararbeit: `Seminararbeit.docx`, Stand 29.07.2026, 16:17:49 Uhr, SHA-256 `8c9e9f235e84a570d8990aa79352a790d93808f8125756a4ea7dc6d3685c1762`
- ausgewählte Samples und Messwerte: `results/qualitative_selection/selected_examples.json`
- Zuordnung der Samples zu den Abbildungen: `results/qualitative_figures/qualitative_figure_manifest.json`
- quantitative Methodenübersicht: `results/qualitative_selection/method_tradeoff_table.md`
- Rohmetadaten der Läufe: `results/final/**/metadata.json`
- finale Abbildungen: `results/qualitative_figures/per_method/`

Die Seminararbeit bleibt für die wissenschaftliche Argumentation maßgeblich. Diese Datei dient als technische Evidenz- und Interpretationshilfe. Sie enthält keine zusätzliche medizinische Bewertung und erweitert die Aussagen der Arbeit nicht um unbelegte Kausalbehauptungen.

## Lesart der Abbildungen

- `prediction` bezeichnet die vom Modell vorhergesagte Ausgangsklasse. Der danebenstehende Wert ist die Konfidenz des Modells für diese Klasse.
- Die Klasse und Konfidenz rechts vom Pfeil beziehen sich auf das Counterfactual.
- `target` ist die vorgegebene Zielklasse.
- `valid: yes` bedeutet ausschließlich, dass die Vorhersage des Modells nach der Veränderung der Zielklasse entspricht. Daraus folgt weder visuelle noch medizinische Plausibilität.
- `MAD (l1_mean)` ist die mittlere absolute Pixelabweichung zwischen Original und Counterfactual.
- `RMS (l2_mean)` ist `sqrt(mean((CF - Original)^2))` im Bildwertebereich `[0, 1]`. Es handelt sich nicht um die unnormalisierte volle L2-Norm.
- Bei CFProto, Goyal-CVE und DVCE ist die `changed pixel fraction` der Pixelanteil oberhalb des jeweils ausgewiesenen Schwellwerts. Für CFProto und Goyal-CVE beträgt dieser `0,03`, für DVCE `0,05`.
- Bei SEDC-T entspricht die `changed pixel fraction` dem von den ausgewählten Segmentmasken bedeckten Bildanteil. Es wird hierfür kein Intensitätsschwellwert verwendet.
- Bei Goyal-CVE ist die `embedding distance` die Kosinusdistanz zwischen Original und ausgewähltem Distraktor im normalisierten penultimativen ResNet18-Merkmalsraum. Die `feature-cell edits` geben die Anzahl der ausgetauschten Zellen des `7 × 7`-Gitters an.
- Bei SEDC-T gibt `overlay (selected segments)` die Zahl der ausgewählten Segmente an.
- Die Differenzkarten verwenden eine feste Skala. Eine nahezu schwarze Differenzkarte steht daher für tatsächlich sehr kleine absolute Änderungen und ist kein Darstellungsfehler.
- Die Rollen „ausgewogener valider Fall“, „höchstkonfidenter valider Fall“, „valider Fall mit geringer visueller Plausibilität“ und „Failure Case“ strukturieren die qualitative Auswahl. Sie sind keine zusätzlichen quantitativen Gesamtstatistiken.

## BUSI

### Bild 5-1: CFProto original-style

Abbildung: [`per_method/busi/cfproto_original_style_bottleneck256.png`](per_method/busi/cfproto_original_style_bottleneck256.png)

| Rolle | Sample | Vorhersage | Ziel | Valide | MAD | RMS | Geänderte Pixel |
| --- | ---: | --- | --- | --- | ---: | ---: | ---: |
| ausgewogen | 14 | normal `0,98` → benign `0,79` | benign | ja | `0,0003` | `0,0115` | `0,10 %` |
| höchste Konfidenz | 11 | normal `0,96` → benign `0,93` | benign | ja | `0,0007` | `0,0159` | `0,23 %` |
| geringe visuelle Plausibilität | 3 | benign `1,00` → normal `0,61` | normal | ja | `0,0782` | `0,1752` | `31,84 %` |
| Failure Case | 13 | normal `1,00` → benign `0,90` | malignant | nein | `0,0013` | `0,0222` | `0,43 %` |

Beim ausgewogenen Sample 14 genügt eine im Differenzbild kaum erkennbare Änderung für den Wechsel von `normal` zu `benign`; ein klar abgegrenztes sonografisches Merkmal ist daraus nicht ableitbar. Sample 3 zeigt das Gegenbeispiel: Trotz erfolgreicher Zielerreichung werden `31,84 %` der Pixel oberhalb des Schwellwerts verändert und es entsteht eine großflächige körnige Texturänderung. Beim Failure Case 13 bleibt die Änderung klein, das Modell wechselt aber zu `benign` statt zur Zielklasse `malignant`. Die Abbildung trennt damit drei unterschiedliche Eigenschaften: geringe Pixelabweichung, Zielerreichung und medizinische Interpretierbarkeit.

### Bild 5-2: Goyal-CVE

Abbildung: [`per_method/busi/goyal_2019_counterfactual_visual_explanations.png`](per_method/busi/goyal_2019_counterfactual_visual_explanations.png)

| Rolle | Sample | Vorhersage | Zellen | Embedding-Distanz | Geänderte Pixel |
| --- | ---: | --- | ---: | ---: | ---: |
| ausgewogen | 6 | malignant `0,98` → benign `0,52` | 7 | `0,2184` | `13,42 %` |
| höchste Konfidenz | 13 | normal `1,00` → malignant `0,63` | 9 | `0,3311` | `16,44 %` |
| geringe visuelle Plausibilität | 9 | malignant `1,00` → benign `0,49` | 26 | `0,2873` | `47,26 %` |

Alle drei Counterfactuals erreichen ihre jeweilige Zielklasse; die verwendeten Distraktoren gehören jeweils zur Zielklasse. Im ausgewogenen Sample 6 reichen sieben Zellen im zentralen Läsionsbereich für den Klassenwechsel aus. Die Counterfactual-Konfidenz von `0,52` zeigt, dass das Ergebnis knapp hinter der Entscheidungsgrenze liegt. Beim höchstkonfidenten Fall sind neun Zellen bereits über mehrere Bildbereiche verteilt. Sample 9 benötigt 26 von 49 Zellen und verändert `47,26 %` der Pixel. Die blockförmigen Übergänge und die großflächige Übernahme aus einem anderen Patientenbild zeigen, dass eine reale Distraktorquelle nicht automatisch eine medizinisch isolierte Läsionsveränderung erzeugt. Die Embedding-Distanz dokumentiert die Ähnlichkeit zum gewählten Distraktor, ist aber kein Plausibilitätsnachweis.

### Bild 5-3: SEDC-T original-style

Abbildung: [`per_method/busi/sedc_t_original_style_best_first.png`](per_method/busi/sedc_t_original_style_best_first.png)

| Rolle | Sample | Vorhersage | Ziel | Valide | Segmente | Maskenanteil |
| --- | ---: | --- | --- | --- | ---: | ---: |
| ausgewogen | 14 | normal `0,98` → benign `0,73` | benign | ja | 2 | `4,43 %` |
| höchste Konfidenz | 0 | benign `1,00` → malignant `0,63` | malignant | ja | 3 | `6,24 %` |
| geringe visuelle Plausibilität | 9 | malignant `1,00` → benign `0,47` | benign | ja | 16 | `53,83 %` |
| Failure Case | 3 | benign `1,00` → malignant `0,98` | normal | nein | 16 | `60,23 %` |

Beim ausgewogenen Sample 14 genügen zwei kleine, räumlich getrennte Segmente und ein Maskenanteil von `4,43 %` für den Klassenwechsel. Sample 9 betrifft dagegen 16 Segmente und mehr als die Hälfte des Bildes. Der Failure Case 3 ist besonders aussagekräftig: Obwohl 16 Segmente beziehungsweise `60,23 %` der Bildfläche ersetzt werden, wird nicht `normal`, sondern `malignant` mit `0,98` vorhergesagt. SEDC-T ermöglicht eine räumliche Zuordnung der veränderten Bereiche; eine große oder anatomisch breite Segmentauswahl garantiert jedoch weder die Zielerreichung noch medizinische Spezifität.

### Bild 5-4: DVCE mit Cone Projection und OpenAI-Checkpoint

Abbildung: [`per_method/busi/dvce_original_style_with_cone_projection_openai_checkpoint.png`](per_method/busi/dvce_original_style_with_cone_projection_openai_checkpoint.png)

| Rolle | Sample | Vorhersage | Ziel | Valide | MAD | RMS | Geänderte Pixel |
| --- | ---: | --- | --- | --- | ---: | ---: | ---: |
| ausgewogen | 3 | benign `1,00` → normal `0,67` | normal | ja | `0,0195` | `0,0284` | `7,04 %` |
| höchste Konfidenz | 14 | normal `0,98` → benign `1,00` | benign | ja | `0,0238` | `0,0377` | `11,47 %` |
| geringe visuelle Plausibilität | 0 | benign `1,00` → malignant `0,96` | malignant | ja | `0,0307` | `0,0452` | `18,45 %` |
| Failure Case | 1 | benign `1,00` → benign `0,60` | normal | nein | `0,0203` | `0,0298` | `8,26 %` |

Die validen Beispiele zeigen eine über größere Bildbereiche verteilte Glättung von Specklemustern und Gewebeschichten. In Sample 0 treten zusätzlich scharf begrenzte farbige beziehungsweise rechteckige Strukturen auf, die keinem eindeutigen sonografischen Befund zugeordnet werden können. Im Failure Case 1 bleibt das Modell trotz sichtbarer globaler Veränderung bei `benign`. Die Abbildung dokumentiert damit den Domain-Mismatch des generischen Diffusionspriors: Die Veränderungen betreffen eher globale Bildstatistik und Textur als eine klar lokalisierte Läsionseigenschaft.

### Bild 5-5: DVCE mit Cone Projection und BUSI-fine-getuntem Checkpoint

Abbildung: [`per_method/busi/dvce_original_style_with_cone_projection_busi_fine_tuned_checkpoint.png`](per_method/busi/dvce_original_style_with_cone_projection_busi_fine_tuned_checkpoint.png)

| Rolle | Sample | Vorhersage | Ziel | Valide | MAD | RMS | Geänderte Pixel |
| --- | ---: | --- | --- | --- | ---: | ---: | ---: |
| ausgewogen | 5 | malignant `0,98` → benign `1,00` | benign | ja | `0,0224` | `0,0337` | `10,73 %` |
| höchste Konfidenz | 8 | malignant `0,94` → benign `1,00` | benign | ja | `0,0262` | `0,0392` | `13,52 %` |
| geringe visuelle Plausibilität | 0 | benign `1,00` → malignant `1,00` | malignant | ja | `0,0349` | `0,0523` | `22,35 %` |

Alle dargestellten Fälle erreichen die Zielklasse mit `1,00`. Gegenüber der generischen Konfiguration in Bild 5-4 bleibt die grundlegende Ultraschallstruktur deutlicher erkennbar. Dennoch entstehen auffällige neue Strukturen: eine ringförmige dunkle Struktur im ausgewogenen Fall, zusätzliche rundliche echoarme Bereiche im zweiten Fall und eine großflächig veränderte Gewebetextur im wenig plausiblen Fall. Die konkrete fine-getunte Konfiguration weist im Gesamtlauf eine höhere Modellvalidität auf als die OpenAI-Cone-Konfiguration. Wegen weiterer Unterschiede der Läufe ist daraus kein isolierter kausaler Fine-Tuning-Effekt ableitbar. Auch hier belegt hohe Modellvalidität keine medizinische Plausibilität.

### Bild 5-6: DVCE ohne Cone Projection und mit BUSI-fine-getuntem Checkpoint

Abbildung: [`per_method/busi/dvce_original_style_without_cone_projection_busi_fine_tuned_checkpoint.png`](per_method/busi/dvce_original_style_without_cone_projection_busi_fine_tuned_checkpoint.png)

| Rolle | Sample | Vorhersage | Ziel | Valide | MAD | RMS | Geänderte Pixel |
| --- | ---: | --- | --- | --- | ---: | ---: | ---: |
| ausgewogen | 5 | malignant `0,98` → benign `1,00` | benign | ja | `0,0215` | `0,0307` | `9,99 %` |
| höchste Konfidenz | 14 | normal `0,98` → benign `1,00` | benign | ja | `0,0245` | `0,0355` | `12,66 %` |
| geringe visuelle Plausibilität | 0 | benign `1,00` → malignant `1,00` | malignant | ja | `0,0325` | `0,0481` | `20,06 %` |

Die Abbildung ist als Ablationsvergleich zu Bild 5-5 zu lesen. Alle drei Counterfactuals sind modellvalide und erreichen `1,00` Konfidenz. Die Änderungen bleiben jedoch nicht eindeutig befundbezogen: Im ausgewogenen Fall wird die Läsionsumgebung diffus umgeformt, beim zweiten Fall tritt eine lokalisierte farbliche Struktur auf, und im wenig plausiblen Fall entsteht eine sternförmige dunkle Struktur bei gleichzeitig veränderter Bildbeschriftung. Die Ablation untersucht die Guidance-Variante; sie stellt keine klinisch realistische Tumortransformation dar.

## Pneumonia

### Bild 5-7: CFProto original-style

Abbildung: [`per_method/pneumonia/cfproto_original_style_bottleneck256.png`](per_method/pneumonia/cfproto_original_style_bottleneck256.png)

| Rolle | Sample | Vorhersage | Ziel | Valide | MAD | RMS | Geänderte Pixel |
| --- | ---: | --- | --- | --- | ---: | ---: | ---: |
| ausgewogen | 3 | NORMAL `0,57` → PNEUMONIA `0,51` | PNEUMONIA | ja | `0,0000` | `0,0017` | `0,00 %`* |
| höchste Konfidenz | 16 | PNEUMONIA `1,00` → NORMAL `0,79` | NORMAL | ja | `0,0028` | `0,0272` | `1,72 %` |
| geringe visuelle Plausibilität | 8 | NORMAL `0,89` → PNEUMONIA `0,76` | PNEUMONIA | ja | `0,0153` | `0,0604` | `10,52 %` |

\* Der ungerundete Anteil bei Sample 3 beträgt `0,002657 %`; die Anzeige `0,00 %` ist die korrekte Rundung auf zwei Nachkommastellen.

Beim ausgewogenen Sample 3 kippt die Vorhersage nur knapp zu `PNEUMONIA`, obwohl die Differenzkarte nahezu schwarz bleibt. Ein konkretes pneumonietypisches Muster ist daraus nicht ableitbar. Bei Sample 16 sind punktförmige Änderungen über Thorax und Randbereiche verteilt. Sample 8 zeigt ein großflächiges raster- und texturartiges Störmuster. Die vollständige Modellvalidität der Methode auf dem untersuchten Pneumonia-Manifest ist deshalb kein Nachweis dafür, dass gezielt medizinisch plausible Pneumonie-Evidenz erzeugt oder entfernt wurde.

### Bild 5-8: Goyal-CVE

Abbildung: [`per_method/pneumonia/goyal_2019_counterfactual_visual_explanations.png`](per_method/pneumonia/goyal_2019_counterfactual_visual_explanations.png)

| Rolle | Sample | Vorhersage | Zellen | Embedding-Distanz | Geänderte Pixel |
| --- | ---: | --- | ---: | ---: | ---: |
| ausgewogen | 3 | NORMAL `0,57` → PNEUMONIA `0,54` | 1 | `0,0684` | `1,97 %` |
| höchste Konfidenz | 1 | NORMAL `0,88` → PNEUMONIA `0,58` | 5 | `0,0872` | `9,95 %` |
| geringe visuelle Plausibilität | 10 | PNEUMONIA `1,00` → NORMAL `0,51` | 32 | `0,4402` | `57,81 %` |

Beim ausgewogenen Sample 3 genügt eine einzige Zelle im unteren Thoraxbereich für einen knappen Klassenwechsel. Der höchstkonfidente Fall benötigt fünf über Lungenbasis, Zwerchfell- und Randbereiche verteilte Zellen. Sample 10 ersetzt dagegen 32 von 49 Zellen beziehungsweise `57,81 %` der Pixel. Das resultierende Mosaik übernimmt großflächig Anatomie- und Aufnahmeunterschiede des Distraktorbildes. Wenige Zell-Swaps können eine räumlich gut lokalisierte Modellerklärung liefern; bei vielen Swaps liegt keine medizinisch isolierte Veränderung mehr vor. Die deutlich größere Embedding-Distanz von Sample 10 ist mit dieser Auswahl vereinbar, ersetzt aber keine visuelle oder medizinische Plausibilitätsprüfung.

### Bild 5-9: SEDC-T original-style

Abbildung: [`per_method/pneumonia/sedc_t_original_style_best_first.png`](per_method/pneumonia/sedc_t_original_style_best_first.png)

| Rolle | Sample | Vorhersage | Ziel | Valide | Segmente | Maskenanteil |
| --- | ---: | --- | --- | --- | ---: | ---: |
| ausgewogen | 3 | NORMAL `0,57` → PNEUMONIA `0,64` | PNEUMONIA | ja | 1 | `6,40 %` |
| höchste Konfidenz | 4 | NORMAL `0,84` → PNEUMONIA `0,63` | PNEUMONIA | ja | 4 | `17,21 %` |
| geringe visuelle Plausibilität | 19 | PNEUMONIA `0,99` → NORMAL `0,54` | NORMAL | ja | 10 | `28,22 %` |
| Failure Case | 17 | PNEUMONIA `1,00` → PNEUMONIA `0,82` | NORMAL | nein | 15 | `44,21 %` |

Beim ausgewogenen Sample 3 liegt das entscheidende Segment am äußeren rechten Bildrand beziehungsweise im Schulterbereich und nicht innerhalb eines klaren Lungenbefunds. Dass die Vorhersage dennoch zu `PNEUMONIA` wechselt, ist ein Hinweis auf ein modellrelevantes, nicht eindeutig pathologisches Bildmerkmal. Die weiteren validen Beispiele benötigen breitere Änderungen. Im Failure Case 17 werden 15 Segmente und `44,21 %` der Bildfläche verändert, das Modell bleibt aber mit `0,82` bei `PNEUMONIA`. Die Abbildung begründet sowohl die Untersuchung einer ROI-Einschränkung als auch die begrenzte Validität der uneingeschränkten Suche.

### Bild 5-10: SEDC-T mit Lung-field-ROI

Abbildung: [`per_method/pneumonia/sedc_t_lung_field_roi_ablation.png`](per_method/pneumonia/sedc_t_lung_field_roi_ablation.png)

| Rolle | Sample | Vorhersage | Ziel | Valide | Segmente | Maskenanteil |
| --- | ---: | --- | --- | --- | ---: | ---: |
| ausgewogen | 3 | NORMAL `0,57` → PNEUMONIA `0,62` | PNEUMONIA | ja | 1 | `3,38 %` |
| höchste Konfidenz | 6 | NORMAL `0,95` → PNEUMONIA `0,69` | PNEUMONIA | ja | 9 | `25,22 %` |
| geringe visuelle Plausibilität | 9 | NORMAL `0,84` → PNEUMONIA `0,55` | PNEUMONIA | ja | 6 | `20,06 %` |
| Failure Case | 19 | PNEUMONIA `0,99` → PNEUMONIA `0,97` | NORMAL | nein | 3 | `5,88 %` |

Beim ausgewogenen Sample 3 konzentriert sich die Änderung auf ein Segment im oberen Lungenfeld. Gegenüber Bild 5-9 ist sie mit `3,38 %` Maskenanteil räumlich fokussierter. Die anderen Beispiele zeigen jedoch die Grenze der geometrischen ROI: Für valide Counterfactuals werden teilweise große Teile beider Lungenfelder oder angrenzender Bereiche ersetzt. Im Failure Case bleibt das Modell trotz drei ausgewählter Segmente mit `0,97` bei `PNEUMONIA`. Die ROI kann die Lokalität einzelner Beispiele verbessern, ist aber keine echte Lungensegmentierung und garantiert weder Zielerreichung noch medizinische Spezifität.

### Bild 5-11: DVCE mit Cone Projection und OpenAI-Checkpoint

Abbildung: [`per_method/pneumonia/dvce_original_style_with_cone_projection_openai_checkpoint.png`](per_method/pneumonia/dvce_original_style_with_cone_projection_openai_checkpoint.png)

| Rolle | Sample | Vorhersage | Ziel | Valide | MAD | RMS | Geänderte Pixel |
| --- | ---: | --- | --- | --- | ---: | ---: | ---: |
| ausgewogen | 17 | PNEUMONIA `1,00` → NORMAL `0,52` | NORMAL | ja | `0,0133` | `0,0204` | `3,07 %` |
| höchste Konfidenz | 7 | NORMAL `0,69` → PNEUMONIA `1,00` | PNEUMONIA | ja | `0,0187` | `0,0267` | `6,23 %` |
| geringe visuelle Plausibilität | 10 | PNEUMONIA `1,00` → NORMAL `0,91` | NORMAL | ja | `0,0191` | `0,0310` | `7,28 %` |
| Failure Case | 13 | PNEUMONIA `1,00` → PNEUMONIA `0,81` | NORMAL | nein | `0,0146` | `0,0266` | `3,48 %` |

Das ausgewogene Sample 17 erreicht `NORMAL` bei nur `3,07 %` geänderten Pixeln oberhalb des Schwellwerts. Das Counterfactual zeigt dennoch eine globale Glättung von Rippen, Lungenzeichnung und Herzsilhouette. Derselbe Rekonstruktionscharakter prägt die anderen validen Fälle. Im Failure Case 13 bleibt die Vorhersage trotz globaler Umformung bei `PNEUMONIA`. Die Abbildung zeigt damit, weshalb der generische Diffusionsprior trotz teilweise kleiner Messwerte und hoher Konfidenzen medizinisch nur eingeschränkt interpretierbar ist.

### Bild 5-12: DVCE mit Cone Projection und Pneumonia-fine-getuntem Checkpoint

Abbildung: [`per_method/pneumonia/dvce_original_style_with_cone_projection_pneumonia_fine_tuned_checkpoint.png`](per_method/pneumonia/dvce_original_style_with_cone_projection_pneumonia_fine_tuned_checkpoint.png)

| Rolle | Sample | Vorhersage | Ziel | Valide | MAD | RMS | Geänderte Pixel |
| --- | ---: | --- | --- | --- | ---: | ---: | ---: |
| ausgewogen | 19 | PNEUMONIA `0,99` → NORMAL `0,99` | NORMAL | ja | `0,0171` | `0,0331` | `4,88 %` |
| höchste Konfidenz | 3 | NORMAL `0,57` → PNEUMONIA `1,00` | PNEUMONIA | ja | `0,0195` | `0,0267` | `6,38 %` |
| geringe visuelle Plausibilität | 10 | PNEUMONIA `1,00` → NORMAL `1,00` | NORMAL | ja | `0,0240` | `0,0391` | `12,08 %` |

Alle drei Fälle sind modellvalide und erreichen Konfidenzen zwischen `0,99` und `1,00`. Gegenüber Bild 5-11 bleiben radiografische Grundstrukturen teilweise schärfer erhalten. Gleichzeitig treten punkt- und linienförmige helle, dunkle oder farbige Strukturen sowie Verformungen an Rippen- und Wirbelsäulenbereichen auf. Diese Artefakte lassen sich nicht eindeutig einem pneumoniespezifischen Befund zuordnen. Die Abbildung dokumentiert die hohe Zielklassenerreichung dieser konkreten Konfiguration, nicht einen nachgewiesenen medizinischen oder isolierten kausalen Fine-Tuning-Effekt.

### Bild 5-13: DVCE ohne Cone Projection und mit OpenAI-Checkpoint

Abbildung: [`per_method/pneumonia/dvce_original_style_without_cone_projection_openai_checkpoint.png`](per_method/pneumonia/dvce_original_style_without_cone_projection_openai_checkpoint.png)

| Rolle | Sample | Vorhersage | Ziel | Valide | MAD | RMS | Geänderte Pixel |
| --- | ---: | --- | --- | --- | ---: | ---: | ---: |
| ausgewogen | 19 | PNEUMONIA `0,99` → NORMAL `1,00` | NORMAL | ja | `0,0148` | `0,0275` | `3,47 %` |
| höchste Konfidenz | 2 | NORMAL `0,74` → PNEUMONIA `1,00` | PNEUMONIA | ja | `0,0194` | `0,0276` | `6,93 %` |
| geringe visuelle Plausibilität | 10 | PNEUMONIA `1,00` → NORMAL `1,00` | NORMAL | ja | `0,0204` | `0,0328` | `8,65 %` |

Alle dargestellten Counterfactuals erreichen die Zielklasse mit `1,00`. Im engen Sinn der Modellvalidität ist diese Ablation damit erfolgreicher als die Cone-Konfiguration mit OpenAI-Checkpoint in Bild 5-11. Visuell bleiben jedoch globale Texturänderungen, anatomische Verschiebungen sowie künstliche punkt- oder linienförmige Strukturen erkennbar. Insbesondere bei den Wechseln von `PNEUMONIA` zu `NORMAL` wird die gesamte Aufnahme umgeformt, ohne dass sich die Entscheidung auf die Entfernung einer klaren fokalen Verschattung zurückführen lässt. Höhere Zielklassenkonfidenz ohne Cone Projection ist daher kein Beleg für bessere medizinische Plausibilität.

### Bild 5-14: DVCE ohne Cone Projection und mit Pneumonia-fine-getuntem Checkpoint

Abbildung: [`per_method/pneumonia/dvce_original_style_without_cone_projection_pneumonia_fine_tuned_checkpoint.png`](per_method/pneumonia/dvce_original_style_without_cone_projection_pneumonia_fine_tuned_checkpoint.png)

| Rolle | Sample | Vorhersage | Ziel | Valide | MAD | RMS | Geänderte Pixel |
| --- | ---: | --- | --- | --- | ---: | ---: | ---: |
| ausgewogen | 17 | PNEUMONIA `1,00` → NORMAL `0,99` | NORMAL | ja | `0,0137` | `0,0232` | `3,00 %` |
| höchste Konfidenz | 1 | NORMAL `0,89` → PNEUMONIA `1,00` | PNEUMONIA | ja | `0,0168` | `0,0258` | `5,23 %` |
| geringe visuelle Plausibilität | 10 | PNEUMONIA `1,00` → NORMAL `0,99` | NORMAL | ja | `0,0195` | `0,0305` | `7,59 %` |

Die drei Counterfactuals sind mit Konfidenzen von `0,99` bis `1,00` modellvalide. Beim ausgewogenen Sample 17 liegt der geänderte Pixelanteil oberhalb des Schwellwerts bei `3,00 %`; dennoch verändern sich Kontrast, Wirbelsäulen- und Lungenzeichnung über größere Bereiche. Die übrigen Fälle zeigen diffuse Texturverstärkungen und anatomische Glättungen, die nicht als eindeutiges Hinzufügen oder Entfernen einer Pneumonie gelesen werden können. Gemeinsam mit Bild 5-12 zeigt die Abbildung, dass Cone und No-Cone unterschiedliche Artefaktmuster erzeugen. Die Wahl der Guidance ist deshalb ein Plausibilitäts- und nicht nur ein Validitätsproblem.

## Methodenübergreifende Schlussfolgerungen

1. **Modellvalidität und medizinische Plausibilität bleiben strikt getrennt.** Ein erfolgreicher Zielklassenwechsel zeigt nur, dass das untersuchte Modell auf die Veränderung reagiert. Er beweist keine realistische Krankheitsentwicklung und keine kausale medizinische Transformation.
2. **Kleine Distanzwerte reichen als Qualitätsnachweis nicht aus.** Besonders CFProto und einzelne DVCE-Fälle zeigen, dass geringe MAD-, RMS- oder Changed-Pixel-Werte mit kaum interpretierbaren, verteilten oder global wirkenden Veränderungen einhergehen können.
3. **Die Lokalität ist methodenabhängig.** Goyal-CVE lokalisiert Änderungen auf einem groben `7 × 7`-Gitter und bindet sie an reale Distraktorbilder. SEDC-T lokalisiert Änderungen über Segmente. Beide Darstellungen sind räumlich nachvollziehbar, können aber breite Anatomie- oder Aufnahmeunterschiede enthalten.
4. **Die Pneumonia-ROI ist eine Ablation, keine medizinische Lungensegmentierung.** Sie kann einzelne Änderungen fokussieren, erhöht in den vorliegenden Ergebnissen aber nicht automatisch die Validität und garantiert keine pneumoniespezifische Evidenz.
5. **DVCE bleibt stark von Checkpoint und Guidance abhängig.** Die fine-getunten und die No-Cone-Konfigurationen erreichen in den untersuchten Läufen häufig eine höhere Zielklassenerreichung. Die Abbildungen zeigen jedoch weiterhin globale Rekonstruktionen und künstliche Strukturen. Aufgrund nicht vollständig kontrollierter Ausführungsbedingungen wird kein isolierter kausaler Fine-Tuning-Effekt behauptet.
6. **Die Abbildungen sind Ergebnisbestandteil, nicht Dekoration.** Jede Abbildung erklärt entweder eine Methode, einen Validitäts-Lokalitäts-Plausibilitäts-Trade-off, einen Failure Case oder eine klar gekennzeichnete Ablation.
