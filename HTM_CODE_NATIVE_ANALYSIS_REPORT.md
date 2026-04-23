# HTM Code-Native: Повний аналіз концепції та кодової бази

Дата аналізу: 2026-04-23
Репозиторій: `E:\htm_project`

## 1. Що саме було проаналізовано

- Концепція: `HTM_Code_Native_Final_Concept.md` (2314 рядків)
- Додаткові документи: `README.md`, `docs/phase_a_overview.md`
- Конфігурація: `configs/phase_a.yaml`, `pyproject.toml`
- Кодова база: 43 Python-файли, приблизно 8348 рядків коду в `src/htm_code_native`
- Тести: 20 Python-файлів, приблизно 1178 рядків у `tests`
- Фактична перевірка:
  - `python -m pytest -q`
  - `python benchmarks/acceptance.py --phase phase_d --repo-root tests/fixtures/repo_graph_workspace --report-path tests/fixtures/repo_graph_workspace/reports/junit.xml --report-path tests/fixtures/repo_graph_workspace/reports/eslint.json --max-steps 3`
  - `python benchmarks/microbench.py tests/fixtures/repo_graph_workspace/app/core.py --repo-root tests/fixtures/repo_graph_workspace --report-path tests/fixtures/repo_graph_workspace/reports/junit.xml --report-path tests/fixtures/repo_graph_workspace/reports/eslint.json`

## 2. Висновок у двох реченнях

Це не "повністю готова фінальна реалізація" всієї архітектури з концепту, а доволі сильний вертикальний прототип, який реально покриває Phase A, B, C, D і частину Phase E, причому багато ключових підсистем уже зшиті в один forward-прохід, smoke-train loop, CLI, benchmark-и і тестовий набір.

Найбільша проблема не у відсутності модулів як таких, а в тому, що кілька критичних підсистем існують лише як локальний внутрішньобатчевий зріз: пам'ять скидається на кожен `forward`, cold semantic lane практично не живе як довготривала пам'ять, exact lanes не дають справжнього byte-level exact reproduction на рівні фінального інференсу, а edit-mode та benchmark/acceptance story суттєво простіші за те, що вимагає фінальна концепція.

## 3. Що вимагає концепція

Фінальний документ задає таку цільову систему:

- Ієрархічне HSSM-ядро з гібридним stride + boundary schedule.
- Семантичну пам'ять hot/cold для стислого long-context представлення.
- Exact Recent Memory для короткої точної пам'яті.
- Exact Episodic Memory для незмінних чанків і pointer-copy.
- Repository Graph Memory для міжфайлових, символічних, тестових і діагностичних зв'язків.
- Retrieval Router з warmup, oracle-guided routing, anti-collapse, energy-aware gating.
- Mixture output: LM + semantic + copy + pointer + graph.
- Фазовий curriculum від bootstrap до full benchmark harness.
- Довготривале, інкрементальне, bounded-compute inference без глобального перегляду всієї історії на кожен токен.

Критично важливо: концепт описує не просто наявність модулів, а саме персистентну інкрементальну memory-архітектуру з реальною фазовою еволюцією, енергетичними бюджетами, maintenance scheduling, acceptance criteria та benchmark-гейтами.

## 4. Що реально є в репозиторії

### 4.1 Загальна архітектура

У коді насправді є такі великі шари:

- `tokenizer/`: tree-sitter based parser/tokenizer + structural enrichment + boundary scheduling
- `data/`: типи, vocabulary registry, featurizer
- `encoders/`: token/byte/structure embedding fusion
- `hssm/`: ієрархічне state-space ядро
- `memory/semantic/`: hot/cold semantic memory
- `memory/exact_recent/`: recent ring + copy distribution
- `memory/exact_episodic/`: chunk store + pointer distribution
- `memory/repo_graph/`: repo indexer + graph retrieval
- `router/`: двоступеневий router з warmup-логікою
- `model/phase_a.py`: головна інтеграція всіх lane-ів
- `training/`: task building, optimizer groups, maintenance scheduling, probes
- `editing/`: окремий planner для edit/patch candidate generation
- `cli/`: tokenize / inspect / run-forward / smoke-train
- `benchmarks/`: acceptance + microbench

### 4.2 Реальний статус по фазах

#### Phase A

Реально реалізовано:

- code-aware token stream
- byte-aligned document model
- AST/symbol enrichment
- boundary scheduling
- code/byte/structure embedding
- HSSM
- semantic hot/cold memory primitives
- semantic + LM fused output
- training loop, hierarchical consistency loss, sparse entropy loss

Висновок: як вертикальний прототип Phase A реалізована добре.

#### Phase B

Реально реалізовано:

- exact recent ring buffer
- recent copy distribution
- recent copy loss
- phase gating у моделі та smoke-train
- router warmup logic для phase_b

Висновок: реалізація є, але exactness тут токенова, а не повноцінно byte-span generative exact lane.

#### Phase C

Реально реалізовано:

- immutable episodic chunks
- chunk metadata
- pointer-style token distribution
- episodic loss
- phase gating

Висновок: Phase C присутня як working slice.

#### Phase D

Реально реалізовано:

- repository graph indexer
- import/call/test/diagnostic/config node ingestion
- graph scoring
- graph prior + copy support
- symbol-link style supervision
- graph-aware phase gating

Висновок: це вже не заглушка, а робочий модуль, але він здебільшого heuristic-heavy і brute-force.

#### Phase E

Офіційні документи репозиторію суперечливі:

- `README.md:48` каже, що router ще відкладений.
- `pyproject.toml:8` взагалі називає репозиторій "Phase A implementation".
- Але в коді реально є `TwoStageRouter` у `src/htm_code_native/router/stub.py:41`, він інтегрований у `src/htm_code_native/model/phase_a.py`, має warmup/oracle/dropout/collapse logic і окремі втрати.

Висновок: Phase E реалізована частково й експериментально, але документація цього не відображає.

#### Phase F

Частково реалізовано:

- acceptance script
- microbenchmark
- phase-exit probes

Не реалізовано повноцінно:

- широкий benchmark harness по всіх acceptance axes з концепту
- long-context sweep
- VRAM / memory footprint reporting
- реальні end-to-end repo-edit acceptance suites
- повноцінні energy proxies у сенсі фінального документа

## 5. Що вже повністю реалізовано

Нижче "повністю" означає: модуль є, він зшитий з рештою системи, має хоча б базове тестове покриття і реально викликається.

### 5.1 Парсинг, токенізація, byte alignment

Повністю реалізовано у межах поточного vertical slice:

- multi-language language detection через suffix mapping
- tree-sitter parsing для Python, JS, TS, JSON, YAML, TOML
- light fallback parser для INI
- byte spans для токенів
- AST node spans
- symbol spans
- token-level structural metadata
- synthetic `INDENT`, `DEDENT`, `NEWLINE` для Python

Ключові файли:

- `src/htm_code_native/tokenizer/tree_sitter_backend.py`
- `src/htm_code_native/tokenizer/python_tokenizer.py`
- `src/htm_code_native/tokenizer/structure.py`
- `src/htm_code_native/tokenizer/boundary.py`

Тести:

- `tests/unit/test_tokenizer_pipeline.py`

### 5.2 Фічеризація batch-у

Повністю реалізовано:

- token ids
- token class ids
- language ids
- scope/file/symbol ids
- byte window tensors
- AST path tensors
- targets
- boundaries tensor map

Ключовий файл:

- `src/htm_code_native/data/featurizer.py`

### 5.3 Embedding pipeline

Повністю реалізовано як fused embedding block:

- token embeddings
- class embeddings
- language embeddings
- scope embeddings
- position embeddings
- byte embeddings + pooling
- AST type/depth embeddings
- symbol/file embeddings

Ключовий файл:

- `src/htm_code_native/encoders/code.py`

### 5.4 HSSM

Повністю реалізовано як робоче ієрархічне ядро:

- multi-level cells
- lower aggregation
- top-down influence
- gated update
- norm clipping
- update masks
- master state

Ключовий файл:

- `src/htm_code_native/hssm/core.py`

Тести:

- `tests/unit/test_hssm_and_memory.py`

### 5.5 Exact Recent Memory

Повністю реалізовано як локальний ring-buffer module:

- writes
- wraparound
- overwrite accounting
- key/query scoring
- attention
- token distribution accumulation

Ключовий файл:

- `src/htm_code_native/memory/exact_recent/stub.py`

Тести:

- `tests/unit/test_exact_recent_phase_b.py`

### 5.6 Exact Episodic Memory

Повністю реалізовано як working slice:

- chunk finalization
- immutable chunk payloads
- pointer keys
- retrieval
- top-k chunk scoring
- pointer distribution

Ключовий файл:

- `src/htm_code_native/memory/exact_episodic/stub.py`

Тести:

- `tests/unit/test_exact_episodic_phase_c.py`

### 5.7 Repository Graph Memory

Повністю реалізовано як vertical slice:

- file/symbol/import/test/diagnostic/config/reference nodes
- import closure
- call edges
- report ingestion from junit/eslint/tsc-like inputs
- graph query context
- graph prior + copy support
- graph stats

Ключові файли:

- `src/htm_code_native/memory/repo_graph/stub.py`
- `tests/unit/test_repo_graph_phase_d.py`

### 5.8 Router module

Повністю реалізовано як окремий модуль:

- pre-router
- post-router
- thresholding
- top-k expensive gating
- warmup interpolation with oracle
- lane dropout
- collapse detector
- entropy diagnostics

Ключовий файл:

- `src/htm_code_native/router/stub.py`

Тести:

- `tests/unit/test_router_warmup.py`

### 5.9 Main fused model

Повністю реалізовано як один інтегрований `forward`:

- encoder -> HSSM -> semantic memory -> ERM -> EEM -> graph -> router -> blended distribution
- phase gating
- task-aware graph gating
- diagnostics
- memory stats
- auxiliary outputs for training

Ключовий файл:

- `src/htm_code_native/model/phase_a.py`

### 5.10 Smoke-train, probes, CLI

Повністю реалізовано:

- CLI-команди
- smoke-train
- eval-only phase probes
- optimizer grouping
- gradient clipping
- maintenance scheduler

Ключові файли:

- `src/htm_code_native/cli/__init__.py`
- `src/htm_code_native/training/*.py`

Тести:

- `tests/integration/test_phase_a_model.py`
- `tests/integration/test_training_harness.py`

## 6. Що реалізовано лише частково

### 6.1 Семантична cold memory як реальна long-term memory

Формально cold memory є.

Практично:

- `src/htm_code_native/model/phase_a.py:154-157` скидає пам'ять на кожен `forward`.
- `src/htm_code_native/model/phase_a.py:264-266` дозволяє cold semantic lane тільки якщо вже існують `cold_clusters`.
- `src/htm_code_native/model/phase_a.py:594-600` запускає consolidation лише якщо явно переданий `maintenance_budget > 0`.

Наслідок:

- у звичайному `forward` cold lane здебільшого не живе як довга пам'ять;
- у benchmark-і це підтверджено фактом: `avg_cold_reads = 0.0`.

Висновок: модуль існує, але ціль концепту "cold semantic memory as persistent long-term lane" не досягнута.

### 6.2 Exact memory як справді exact code reproduction

ERM та EEM зберігають байтові payload-и, але фінальний model output працює з token-id distributions, а не з повноцінним byte-span copy head.

Наслідок:

- exactness тут переважно на рівні "можемо підняти ймовірність правильного токена";
- це не те саме, що lossless byte-exact regeneration з концепту.

### 6.3 Router як повноцінна навчальна політика

Router написаний якісно як модуль, але його інтегрований життєвий цикл неповний:

- `src/htm_code_native/model/phase_a.py:157` викликає `self.router.reset()` на кожен `forward`.
- отже collapse history не живе між кроками тренування;
- warmup logic у router існує, але накопичувальна anti-collapse поведінка в інтегрованій моделі сильно ослаблена.

Висновок: router є, але його stateful training protocol реалізований неповно.

### 6.4 Edit mode

Є окремий planner:

- `src/htm_code_native/editing/planner.py`

Він уміє:

- ранжувати edit spans,
- пропонувати replacement terms,
- будувати patch candidates,
- робити валідацію через parse/AST,
- рендерити unified diff.

Але:

- у CLI немає окремої команди для цього workflow;
- edit losses (`edit_span_loss`, `edit_patch_loss`, `diagnostic_alignment_loss`) визначені, але за пошуком коду використовуються лише в `losses/core.py`, а не в training loop;
- edit training у `build_task_batch()` зводиться до простого synthetic single-token replacement supervision.

Висновок: edit subsystem існує як окремий експериментальний інструмент, але не є повністю впровадженим mode з концепту.

### 6.5 Benchmark harness

Є:

- acceptance script
- microbench
- phase exit probes

Немає:

- повного benchmark family з концепту
- реальної довгої long-context evaluation
- явних VRAM/peak memory measurements
- нормального real-world repo edit success harness

## 7. Що ще не реалізовано

### 7.1 Персистентне інкрементальне inference між викликами

Це одна з центральних вимог концепту, але модель фактично працює як batch-local simulator.

Не реалізовано повноцінно:

- збереження semantic hot/cold memory між викликами
- збереження ERM між викликами
- збереження router state/history між викликами
- нормальний streaming session-level inference path

EEM частково може переживати кілька `forward`, якщо `reset_eem=False`, але це виняток, а не загальний режим для всієї архітектури.

### 7.2 Search-tree / indexed retrieval для semantic cold memory

Концепт вимагає tree-index / candidate search.

У коді cold memory:

- просто тримає список кластерів;
- рахує score по всіх кластерах;
- робить `topk`.

Це working prototype, але не концептуальний target.

### 7.3 Справжній bounded-compute repo retrieval

`RepositoryGraphMemory.query()` проходить по всіх `index.nodes` на кожен токен і рахує score для кожного кандидата.

Це означає:

- немає ієрархічного graph index search;
- немає cheap candidate pruning поза brute-force;
- complexity target з концепту не досягнутий.

### 7.4 Повноцінний tokenizer training/bootstrap phase

Концепт описує bootstrap / representation and indexing phase.

У коді:

- є `VocabularyRegistry`;
- немає стійкого pretrained tokenizer/vocabulary pipeline;
- немає окремого persistent tokenizer training/fitting stage.

### 7.5 Повний acceptance gate за концептом

Не реалізовано:

- системна перевірка against strong Transformer baseline
- довгий stability sweep
- репозиторні fix benchmarks
- acceptance rule "must win on a real axis"

## 8. Що не так: конкретні проблеми та розриви

### 8.1 Поточний тестовий стан не зелений

Результат запуску:

- `python -m pytest -q`
- Підсумок: `37 passed, 1 failed`

Падає:

- `tests/unit/test_training_tasks.py::test_repo_graph_examples_include_probe_kinds`

Причина:

- `src/htm_code_native/training/tasks.py:188-251` створює `REPO_GRAPH` приклади лише з `definition_use` і двома копіями `diagnostic_to_symbol`.
- `edit_fix` додається тільки в `TaskLabel.EDIT_FIX`, а не в `TaskLabel.REPO_GRAPH`.
- Це суперечить очікуванню тесту в `tests/unit/test_training_tasks.py:27-30`.

Це реальний, не теоретичний дефект.

### 8.2 Документація та код суперечать одне одному

Факти:

- `README.md:48-50` каже, що learned router ще deferred.
- `pyproject.toml:8` називає пакет "Phase A implementation".
- `configs/phase_a.yaml:56` вже ставить `training_phase: phase_d`.
- `src/htm_code_native/router/stub.py:41` містить реальний `TwoStageRouter`.
- `src/htm_code_native/model/phase_a.py` реально використовує router у головному forward.

Висновок:

- реальний стан репозиторію просунутіший за документацію;
- документація застаріла і вводить в оману щодо scope-а.

### 8.3 Найважливіший архітектурний розрив: пам'ять скидається на кожен forward

Факти:

- `src/htm_code_native/model/phase_a.py:154` `self.semantic_memory.reset()`
- `src/htm_code_native/model/phase_a.py:155` `self.exact_recent_memory.reset()`
- `src/htm_code_native/model/phase_a.py:157` `self.router.reset()`
- `src/htm_code_native/model/phase_a.py:159` умовно скидає EEM

Наслідки:

- немає session-level persistence;
- cold semantic memory не накопичується між прогоном задач;
- ERM не є справжньою recent memory across steps of a real session;
- anti-collapse router history стирається на кожному виклику.

Це найбільший conceptual mismatch у всьому репозиторії.

### 8.4 Cold semantic lane фактично майже не живе

Cold lane стає доступною тільки якщо вже існують кластери:

- `src/htm_code_native/model/phase_a.py:264-266`

А кластери з'являються лише через maintenance:

- `src/htm_code_native/model/phase_a.py:594-600`

Оскільки пам'ять скидається на початку `forward`, cold memory не встигає стати повноцінним long-term lane.

Це прямо підтверджено microbenchmark-ом:

- `avg_cold_reads: 0.0`

### 8.5 Exact lanes не дотягують до концептуального "lossless exact archive"

Є raw bytes у слотах/чанках, але:

- read path повертає token distribution;
- вихід моделі змішує token distributions;
- немає окремого byte-level exact emission path у фінальній генерації.

Отже exact lanes більше схожі на token-copy prior, ніж на завершену lossless exact memory system.

### 8.6 Edit training і edit planner не зшиті докупи

Факти:

- `src/htm_code_native/editing/planner.py` містить доволі великий planner.
- `src/htm_code_native/losses/core.py:138-159` містить edit losses.
- Але пошук використання показує, що `edit_span_loss`, `edit_patch_loss`, `diagnostic_alignment_loss` не беруть участі в smoke-train loop.

Наслідок:

- edit mode є як окремий експериментальний інструмент;
- train-time підтримка edit-specific behavior неповна;
- репозиторій ще не дійшов до повноцінного "edit / refactor mode" з концепту.

### 8.7 Repo graph retrieval — робочий, але дорогий і heuristic-heavy

Плюси:

- мультиджерельне наповнення graph index
- імпорти, call edges, tests, diagnostics
- bias-и samefile/import/symbol/test/diagnostic

Мінуси:

- scoring по всіх вузлах на кожен токен
- багато heuristic parsing для TS/JS/config
- немає дешевого indexed retrieval
- symbol-link acceptance зараз слабкий

Фактичне підтвердження:

- acceptance benchmark провалюється на `symbol_link_below_threshold`
- метрика `symbol_link_hit_rate` = `0.0`

### 8.8 Навіть acceptance benchmark не проходить повністю

Результат:

- `phase = phase_d`
- `passed = false`
- `failing_checks = ["symbol_link_below_threshold"]`

Ключові метрики:

- `recent_copy_hit_rate = 0.5543`
- `episodic_hit_rate = 0.3218`
- `graph_copy_hit_rate = 0.0`
- `symbol_link_hit_rate = 0.0`
- `route_entropy = 1.3049`
- `energy_proxy = 4.3708`
- `tokens_per_sec = 21.5570`

Інтерпретація:

- recent copy вже реально працює;
- episodic retrieval теж має сигнал;
- graph lane як symbol-link reasoning ще не доведена до робочого acceptance рівня.

## 9. Детальний аналіз по модулях

### 9.1 `tokenizer/tree_sitter_backend.py`

Сильні сторони:

- найбільш функціонально насичений tokenizer-модуль у репозиторії;
- тримає реальний multi-language ingestion;
- будує `AlignedDocument`, `ParseDocument`, `TokenStructureInfo`, `SyntaxStateFeatures`;
- дає AST/symbol alignment.

Слабкі сторони:

- `PythonTokenizer` і `PythonStructureExtractor` лише thin wrapper-и;
- `_build_structure_views()` проходить named nodes і для кожного named node проходить усі токени в span-matching стилі, тобто має важку складність;
- це хороший vertical slice, але не оптимізована production token pipeline.

### 9.2 `data/vocabulary.py`

Сильні сторони:

- простий і зрозумілий stable registry;
- є `snapshot`, `unk`, `pad`, `boundary`.

Слабкі сторони:

- vocabulary локальна й динамічна, не схожа на окремо навчену/зашиту tokenizer систему;
- немає persistence layer для tokenizer bootstrap phase з концепту.

### 9.3 `hssm/core.py`

Сильні сторони:

- архітектурно це один із найчистіших модулів;
- добре читається;
- directly відповідає частині концепту.

Слабкі сторони:

- це компактна GRU-подібна інтерпретація концепту, а не багата реалізація з окремими maintenance/event policies;
- `last_update_indices` збираються, але майже не відіграють ролі в інтегрованому forward.

### 9.4 `memory/semantic/store.py`

Сильні сторони:

- hot/cold розділення реалізоване;
- є read/write/consolidation;
- є entropy stats.

Слабкі сторони:

- немає tree index для cold search;
- cold clusters не живуть між викликами моделі;
- maintenance policy спрощена до локального бюджетного тригера.

### 9.5 `memory/exact_recent/stub.py`

Сильні сторони:

- реалізація акуратна;
- тести добрі;
- статистика читань/перезаписів прозора.

Слабкі сторони:

- payload bytes зберігаються, але фактично не керують фінальним exact byte-level emission;
- немає persistence across forward calls.

### 9.6 `memory/exact_episodic/stub.py`

Сильні сторони:

- good immutable chunk idea;
- чітка metadata-модель;
- pointer retrieval працює.

Слабкі сторони:

- retrieval все ще token-centric;
- немає окремого зовнішнього index/search layer;
- для повного концепту замало exact edit/copy machinery.

### 9.7 `memory/repo_graph/stub.py`

Сильні сторони:

- дуже змістовний і корисний модуль;
- фактично це найбільший phase_d asset;
- є ingestion діагностик і тестів, а не тільки кодових символів.

Слабкі сторони:

- дуже heuristic-heavy;
- brute-force candidate scoring;
- symbol-link performance поки слабка.

### 9.8 `router/stub.py`

Сильні сторони:

- surprisingly сильна реалізація для "stub"-модуля;
- має теплий старт, oracle mixing, entropy, collapse logic.

Слабкі сторони:

- інтеграційний життєвий цикл зламаний reset-ом у моделі;
- `README` невірно описує статус цього модуля.

### 9.9 `model/phase_a.py`

Це головний вузол репозиторію.

Сильні сторони:

- тут реально відчувається цілісна система, а не набір окремих заглушок;
- phase policies є;
- task-aware graph gating є;
- diagnostics і auxiliary outputs багаті;
- тестове покриття непогане.

Слабкі сторони:

- назва файла історично застаріла, бо модель давно вже не лише Phase A;
- надто багато відповідальності в одному класі;
- lifecycle memory/router state не відповідає фінальному концепту.

### 9.10 `training/tasks.py`

Сильні сторони:

- good task abstraction;
- є AR, INFILL, RECENT_COPY, EPISODIC_RECALL, REPO_GRAPH, EDIT_FIX;
- є phase task weights.

Слабкі сторони:

- реальна помилка з `default_task_examples()`;
- synthetic edit supervision досить груба;
- duplicate `diagnostic_to_symbol` приклад виглядає як явний copy/paste bug.

### 9.11 `cli/__init__.py`

Сильні сторони:

- зручно оглянути стан моделі без додаткової обв'язки;
- smoke-train дає відчуття end-to-end workflow;
- eval-only probes є.

Слабкі сторони:

- CLI не відкриває edit planner як повноцінний режим;
- training loop не використовує edit-specific losses;
- назви команд і моделі ще несуть legacy "phase_a" naming.

## 10. Фактичний тестовий та benchmark статус

### 10.1 Pytest

Команда:

```bash
python -m pytest -q
```

Результат:

- 37 тестів пройшло
- 1 тест упав

Оцінка:

- для експериментального research-style репозиторію це хороший сигнал;
- але сказати "все повністю реалізовано" не можна, бо навіть контрольний тестовий набір не зелений.

### 10.2 Acceptance

Команда:

```bash
python benchmarks/acceptance.py --phase phase_d --repo-root tests/fixtures/repo_graph_workspace --report-path tests/fixtures/repo_graph_workspace/reports/junit.xml --report-path tests/fixtures/repo_graph_workspace/reports/eslint.json --max-steps 3
```

Результат:

- `passed = false`
- провал лише на `symbol_link_below_threshold`

Оцінка:

- repo-graph lane частково працює;
- acceptance для symbol reasoning ще не закрита.

### 10.3 Microbench

Команда:

```bash
python benchmarks/microbench.py tests/fixtures/repo_graph_workspace/app/core.py --repo-root tests/fixtures/repo_graph_workspace --report-path tests/fixtures/repo_graph_workspace/reports/junit.xml --report-path tests/fixtures/repo_graph_workspace/reports/eslint.json
```

Ключові результати:

- `tokens_per_sec = 34.02`
- `avg_hot_reads = 14688.0`
- `avg_cold_reads = 0.0`
- `avg_graph_reads = 124.0`
- `graph_symbol_recall = 0.0`
- `avg_route_entropy = 1.4505`
- `avg_energy_proxy = 4.2903`
- `hard_gated_energy_savings = 5.7097`

Оцінка:

- route gating реально дає savings;
- cold semantic lane не живе;
- graph symbol recall нульова;
- throughput для такого research prototype нормальна, але не доводить досягнення concept-level efficiency targets.

## 11. Матриця статусів

### 11.1 Реально реалізовано добре

- Tokenization + alignment
- AST/symbol enrichment
- Boundary scheduler
- HSSM
- Semantic hot memory
- ERM
- EEM
- Repo graph indexing
- Repo graph retrieval
- Router module
- Phase-aware forward
- Smoke-train CLI
- Probe/eval scripts

### 11.2 Реалізовано, але не доведено до концептуального рівня

- Semantic cold memory
- Router lifecycle in training
- Exact copy semantics
- Episodic pointer semantics
- Graph symbol reasoning
- Edit mode
- Benchmark harness
- Maintenance scheduling
- Energy-aware inference policy

### 11.3 Не реалізовано або не закрито

- Persistent cross-forward memory state
- True streaming inference session mode
- Full Phase F acceptance system
- Strong baseline comparison
- Full exact byte/span generation path
- Повноцінний repo edit mode у CLI/train/eval

## 12. Найважливіші рекомендації по пріоритету

### P0

- Прибрати unconditional reset для semantic/ERM/router state у `PhaseACodeModel.forward()`, або винести це у явний session lifecycle API.
- Виправити `default_task_examples()` так, щоб тестовий набір знову став зелений.
- Привести `README.md` і `pyproject.toml` у відповідність до фактичного стану репозиторію.

### P1

- Зробити cold semantic lane реально життєздатною між forward calls.
- Не скидати router history між optimizer steps.
- Підняти symbol-link performance до проходження acceptance.

### P2

- Підключити edit losses до training loop.
- Додати CLI entrypoint для edit planner.
- Винести repo graph retrieval з brute-force режиму в більш дешевий candidate search.

### P3

- Розділити `model/phase_a.py` на менші модулі.
- Перейменувати legacy naming (`phase_a.py`) у щось, що відповідає реальному scope.
- Додати повноцінний benchmark suite з концепту.

## 13. Підсумковий вердикт

### Якщо оцінювати чесно

Репозиторій уже далеко не на рівні "скелетон" або "заглушки". Це серйозний research-style prototype з реальним кодом, тестами, router-ом, repo graph lane, episodic memory, smoke-train loop і benchmark scripts.

### Але якщо оцінювати саме проти фінальної концепції

Повністю реалізованою фінальну архітектуру назвати не можна.

Правильніше сформулювати так:

- Phase A-D vertical slice реалізований добре.
- Phase E реалізована частково і вже існує в коді, хоча документація це приховує.
- Найбільші conceptual gaps: persistence, cold memory, exact byte-level recall, symbol-link effectiveness, повний edit/benchmark lifecycle.

### Підсумкова оцінка

- Як кодова база-дослідження: сильна
- Як прототип архітектури: переконлива
- Як "повністю завершена реалізація фінального концепту": ні
- Як база для наступного етапу доробки: дуже придатна

