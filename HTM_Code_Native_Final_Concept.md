# HTM Code-Native Final Concept
## Повна консолідована архітектурна специфікація

**Версія:** 2.0 — Final Consolidated Specification  
**Статус:** Source of truth  
**Призначення:** Code-native заміна Transformer для програмування, репозиторних задач, довгого контексту, точного відтворення коду та енергоефективного інкрементального inference.  
**Базове ядро:** Hierarchical State-Space Machine (HSSM) × Semantic Tensor Memory (TRAM/SHM)  
**Розширення:** Exact Recent Memory (ERM) + Exact Episodic Memory (EEM) + Repository Graph Memory (RGM) + Retrieval Router + Copy/Pointer Heads

---

## 0. Статус документа

Цей файл є **однією повною специфікацією**. Він:

- **не скасовує** базову HTM-ідею;
- **не замінює** HSSM × TRAM чимось іншим;
- а **розширює** її до повної code-native memory architecture.

Тобто фінальна архітектура:

$$
\mathrm{HTM\text{-}Code\text{-}Final}
=
\mathrm{HSSM}
\times
\mathrm{Semantic\ Memory}
\times
\mathrm{Exact\ Recent\ Memory}
\times
\mathrm{Exact\ Episodic\ Memory}
\times
\mathrm{Repo\ Graph\ Memory}
\times
\mathrm{Retrieval\ Router}
\times
\mathrm{Copy/Pointer\ Output}
$$

Це **не нова чужа модель**, а **повністю допрацьований HTM**.

---

## 1. Ціль архітектури

Фінальна модель повинна бути оптимізована **під програмування** і в ідеалі вміти:

1. точно відтворювати кодові фрагменти, імена, рядки, числа, шляхи, regex, JSON;
2. стабільно працювати на довгому контексті і не втрачати старі символи/визначення;
3. мислити по ієрархії коду:
   - token
   - expression
   - statement
   - block
   - function/class
   - file/module
   - repo/task
4. бути ефективною на inference:
   - bounded hot path
   - амортизована maintenance
   - без глобального перегляду всієї історії на кожен токен;
5. бути енергоефективною:
   - більшість токенів мають оброблятись локально;
   - важкі memory-операції мають запускатись тільки тоді, коли вони реально корисні.

---

## 2. Негативні вимоги

Ця архітектура **не повинна**:

1. бути pure byte-model як єдиний main stream;
2. бути char-level моделлю;
3. покладати exact recall на semantic compressed memory;
4. компресувати або merge-ити raw code spans, якщо потрібне дослівне відтворення;
5. рахувати дорогі diagnostic regularizers на кожному train step;
6. робити повний dense scan всієї long-term memory на кожному токені.

---

## 3. Представлення вхідних даних

## 3.1 Загальна політика представлення

Для програмування використовується **три потоки даних**:

1. **Основний semantic stream**: code-aware tokens  
2. **Exact stream**: raw UTF-8 bytes  
3. **Structure stream**: parser / AST / symbol / scope metadata

Pure UTF-8 **не використовується як єдиний основний stream**.  
Pure subword без byte-alignment теж **не підходить**, бо втрачає exactness.

Отже політика така:

- **мислення/семантика**: code-aware token stream
- **точна пам’ять**: UTF-8 bytes
- **структурне мислення**: AST / symbol / repo graph

---

## 3.2 Сирий байтовий потік

Нехай вхідний файл або concatenated project context представлений байтовою послідовністю:

$$
B = (b_1, b_2, \dots, b_{n_b}), \quad b_i \in \{0,1,\dots,255\}
$$

Це є **lossless canonical source**.

Усі exact memory lanes зберігають саме байти або byte-aligned spans.

---

## 3.3 Code-aware token stream

Нехай tokenizer перетворює байтовий потік у code-aware token sequence:

$$
T = \mathrm{Tok}(B) = (t_1, t_2, \dots, t_{n_t})
$$

Кожен токен має:

$$
t_j = (c_j, v_j, s_j, e_j, \ell_j, \sigma_j)
$$

де:

- $c_j$ — token class:
  - keyword
  - identifier
  - operator
  - delimiter
  - string
  - number
  - newline
  - indent
  - dedent
  - comment
  - whitespace-control
  - fallback-byte-piece
- $v_j$ — token value/id у словнику
- $[s_j, e_j)$ — byte span токена у сирому потоці
- $\ell_j$ — language id / file language
- $\sigma_j$ — local structural tags

Рекомендований tokenizer:

- lexer-aware
- language-aware
- byte fallback для rare / OOV / weird identifiers
- збереження byte alignment для кожного токена

---

## 3.4 Alignment map

Потрібне явне відображення:

$$
A_T(j) = [s_j, e_j)
$$

тобто кожен token index `j` знає, який byte span він покриває.

Також кожен AST node має:

$$
A_{\mathrm{ast}}(u) = [\hat s_u, \hat e_u)
$$

І кожен symbol має:

$$
A_{\mathrm{sym}}(q) = [\bar s_q, \bar e_q)
$$

Це критично для:

- exact copy
- exact edit
- symbol linking
- patch generation
- span-level retrieval

---

## 3.5 Token embedding

Кожен code token має базовий embedding:

$$
e_j^{(\mathrm{tok})}
=
E_{\mathrm{tok}}[v_j]
+
E_{\mathrm{cls}}[c_j]
+
E_{\mathrm{lang}}[\ell_j]
+
E_{\mathrm{scope}}[\sigma_j]
+
E_{\mathrm{pos}}[p_j]
$$

де:

- $E_{\mathrm{tok}}$ — value embedding
- $E_{\mathrm{cls}}$ — token-class embedding
- $E_{\mathrm{lang}}$ — language embedding
- $E_{\mathrm{scope}}$ — lightweight structure tag embedding
- $E_{\mathrm{pos}}$ — relative/segment positional embedding

---

## 3.6 Byte-span embedding

Для токена `j`, що покриває bytes $b_{s_j:e_j-1}$, обчислюється byte-summary:

$$
u_j^{(\mathrm{byte})}
=
\mathrm{Pool}\left(
E_b[b_{s_j}] + P_b(1),\;
E_b[b_{s_j+1}] + P_b(2),\;
\dots,\;
E_b[b_{e_j-1}] + P_b(e_j-s_j)
\right)
$$

де:

- $E_b$ — byte embedding table розміру $256 \times d_b$
- $P_b$ — byte-local positional encoding
- $\mathrm{Pool}$ — mean/max/attention-pooling

---

## 3.7 Structural embedding

Для токена `j` з AST path:

$$
\Pi_j = (a_{j,1}, a_{j,2}, \dots, a_{j,m_j})
$$

структурний embedding:

$$
u_j^{(\mathrm{struct})}
=
\sum_{r=1}^{m_j}
\left(
E_{\mathrm{ast}}[\mathrm{type}(a_{j,r})]
+
E_{\mathrm{depth}}[r]
\right)
+
E_{\mathrm{sym}}[\mathrm{sym}(j)]
+
E_{\mathrm{file}}[\mathrm{file}(j)]
$$

---

## 3.8 Фінальний вхідний embedding

Фінальний input token embedding:

$$
e_j
=
W_{\mathrm{tok}} e_j^{(\mathrm{tok})}
+
W_{\mathrm{byte}} u_j^{(\mathrm{byte})}
+
W_{\mathrm{struct}} u_j^{(\mathrm{struct})}
+
b_e
$$

Це і є вхід на рівень $l=0$.

---

## 4. Code-native ієрархія рівнів

Рівні повинні відповідати структурі коду, а не лише часовим масштабам.

Рекомендована семантика рівнів:

- **Level 0**: lexer/code tokens
- **Level 1**: expressions / statements
- **Level 2**: control-flow blocks / logical blocks
- **Level 3**: functions / methods / classes
- **Level 4**: file / module
- **Level 5**: repo / task / session memory

Нехай $L$ — загальна кількість рівнів.

---

## 5. Гібридний графік оновлення HSSM

Базовий HTM використовує:

$$
\tau_l = k^l
$$

Для code-native моделі цього недостатньо. Потрібен **гібридний schedule**:

$$
m_l^{(t)} = \mathbf{1}[t \equiv 0 \pmod{\tau_l}] \;\lor\; \mathbf{1}[\mathrm{Boundary}_l(t)=1]
$$

де $\mathrm{Boundary}_l(t)$ — подія структурного завершення для рівня `l`.

Приклади:

- Level 1 boundary:
  - newline with statement end
  - `;`
  - end of expression
- Level 2 boundary:
  - end of block
  - dedent
  - end of `if/for/while/try`
- Level 3 boundary:
  - end of function/class
- Level 4 boundary:
  - end of file/module chunk

Тобто рівень оновлюється:

- або по stride
- або по структурній події

Це робить HTM природним для коду.

---

## 6. HSSM: основні рівняння

## 6.1 Bottom-up aggregation

Для рівня $l>0$, на кроці $t$ формується множина нижчих станів:

$$
\Omega_l(t)
=
\{i \mid i \in \text{current segment for level } l\}
\cup
\{t-\tau_l+1,\dots,t\}
$$

Практична політика:

- використовувати поточний структурний сегмент, якщо він існує;
- інакше fallback до останніх $\tau_l$ станів.

Тоді:

$$
\bar s_{l-1}^{(t)}
=
\frac{1}{|\Omega_l(t)|}
\sum_{i \in \Omega_l(t)} s_{l-1}^{(i)}
$$

Bottom-up проєкція:

$$
h_{l-1}^{(t)}
=
W_l^{(\mathrm{up})}\,\mathrm{LN}(\bar s_{l-1}^{(t)}) + b_l^{(\mathrm{up})}
$$

Для $l=0$:

$$
h_{-1}^{(t)} = e_t
$$

---

## 6.2 Top-down modulation

Для $l<L$:

$$
d_l^{(t)}
=
W_l^{(\mathrm{down})} s_{l+1}^{(\pi_{l+1}(t))} + b_l^{(\mathrm{down})}
$$

де $\pi_{l+1}(t)$ — індекс останнього валідного оновлення рівня $l+1$.

Для $l=L$:

$$
d_L^{(t)} = 0
$$

---

## 6.3 Gated update

Вхідний вектор:

$$
u_l^{(t)} = [h_{l-1}^{(t)};\; d_l^{(t)};\; s_l^{(t-1)}]
$$

Update gate:

$$
z_l^{(t)} = \sigma(W_l^{(z)}u_l^{(t)} + b_l^{(z)})
$$

Reset gate:

$$
r_l^{(t)} = \sigma(W_l^{(r)}u_l^{(t)} + b_l^{(r)})
$$

Candidate state:

$$
\tilde s_l^{(t)}
=
\tanh\left(
W_l^{(s)}
[h_{l-1}^{(t)};\; d_l^{(t)};\; r_l^{(t)} \odot s_l^{(t-1)}]
+
b_l^{(s)}
\right)
$$

If $m_l^{(t)} = 1$:

$$
s_l^{(t)}
=
(1-z_l^{(t)}) \odot s_l^{(t-1)}
+
z_l^{(t)} \odot \tilde s_l^{(t)}
$$

If $m_l^{(t)} = 0$ and $l>0$:

$$
s_l^{(t)} = s_l^{(t-1)}
$$

---

## 6.4 State norm projection

Щоб уникнути drift:

$$
s_l^{(t)} \leftarrow
\begin{cases}
\tau_{\max}\dfrac{s_l^{(t)}}{\|s_l^{(t)}\|_2}, & \|s_l^{(t)}\|_2 > \tau_{\max} \\
s_l^{(t)}, & \text{інакше}
\end{cases}
$$

---

## 6.5 Master state

Фінальний global state:

$$
s_{\mathrm{master}}^{(t)}
=
[s_0^{(t)};\; s_1^{(\pi_1(t))};\; \dots;\; s_L^{(\pi_L(t))}]
$$

---

## 7. Semantic Hierarchical Memory (SHM / TRAM)

SHM — це memory lane для **semantic abstraction**, а не для lossless recall.

Кожен рівень `l` має пам’ять:

$$
\mathcal{M}_l = \mathcal{M}_l^{(\mathrm{hot})} \cup \mathcal{M}_l^{(\mathrm{cold})}
$$

---

## 7.1 Hot memory

Hot memory містить останні semantic slots:

$$
\mathcal{M}_l^{(\mathrm{hot})}
=
\{(k_{l,i}, v_{l,i}, a_{l,i}, \tau_{l,i})\}_{i=1}^{N_l^{(\mathrm{hot})}}
$$

де:

- $k_{l,i}$ — key
- $v_{l,i}$ — value
- $a_{l,i}$ — access statistics
- $\tau_{l,i}$ — write timestamp

Write:

$$
k_l^{(t)} = W_l^{(k)} s_l^{(t)} + b_l^{(k)}
$$
$$
v_l^{(t)} = W_l^{(v)} s_l^{(t)} + b_l^{(v)}
$$

---

## 7.2 Cold memory

Cold memory містить compressed semantic bank:

$$
\mathcal{M}_l^{(\mathrm{cold})}
=
\{(\hat k_{l,j}, \hat v_{l,j}, \hat a_{l,j})\}_{j=1}^{N_l^{(\mathrm{cold})}}
$$

Raw semantic vectors можуть зберігатись у dense cache, а compressed representation — як storage/maintenance form.

---

## 7.3 Query

Semantic query:

$$
q_l^{(t)} = W_l^{(q)} s_{\mathrm{master}}^{(t)} + b_l^{(q)}
$$

---

## 7.4 Hot read

Для hot candidates:

$$
\alpha_{l,i}^{(\mathrm{hot},t)}
=
\mathrm{softmax}_i
\left(
\frac{\langle q_l^{(t)}, k_{l,i}\rangle}{\sqrt{d_k}}
\right)
$$

---

## 7.5 Cold read via search tree

Cold slots організовані через tree index.

Для вузла дерева з центроїдом $\mu_n$:

$$
\mathrm{score}_n^{(t)}
=
\frac{\langle q_l^{(t)}, \mu_n\rangle}{\|q_l^{(t)}\|\,\|\mu_n\|}
$$

На leaf selection отримується невеликий candidate set:

$$
\mathcal{C}_l^{(\mathrm{cold},t)} = \mathrm{LeafSearch}(q_l^{(t)})
$$

Далі точний read серед кандидатів:

$$
\alpha_{l,j}^{(\mathrm{cold},t)}
=
\mathrm{softmax}_j
\left(
\frac{\langle q_l^{(t)}, \hat k_{l,j}\rangle}{\sqrt{d_k}}
\right)
$$

---

## 7.6 Semantic output

Об’єднаний semantic output рівня:

$$
o_l^{(t)}
=
\sum_{i \in \mathcal{C}_l^{(\mathrm{hot},t)}}
\alpha_{l,i}^{(\mathrm{hot},t)} v_{l,i}
+
\sum_{j \in \mathcal{C}_l^{(\mathrm{cold},t)}}
\alpha_{l,j}^{(\mathrm{cold},t)} \hat v_{l,j}
$$

---

## 7.7 Importance and eviction

Важливість hot slot:

$$
\mathrm{Imp}_{l,i}^{(t)}
=
\sum_{\tau=t-T_{\mathrm{win}}+1}^{t}
w_{l,i}^{(\tau)}
$$

Eviction:

$$
i_{\mathrm{evict}}
=
\arg\min_i \mathrm{Imp}_{l,i}^{(t)}
$$

---

## 7.8 Consolidation

Consolidation запускається тільки коли:

$$
\chi_l^{(t)}
=
\mathbf{1}
\left[
\mathrm{fill}_l^{(t)} > \theta_{\mathrm{fill}}
\;\lor\;
\mathrm{debt}_l^{(t)} > \theta_{\mathrm{debt}}
\right]
$$

і бюджет maintenance дозволяє:

$$
\beta_{\mathrm{maint}}^{(t)} > \theta_{\mathrm{maint}}
$$

Semantic clusters:

$$
\mathcal{G}_{l,m}^{(t)} = \mathrm{Cluster}\left(\{k_{l,i}\}\right)
$$

Merged centroid:

$$
\hat k_{l,m}
=
\frac{\sum_{i \in \mathcal{G}_{l,m}} \omega_i k_{l,i}}{\sum_{i \in \mathcal{G}_{l,m}} \omega_i}
$$

$$
\hat v_{l,m}
=
\frac{\sum_{i \in \mathcal{G}_{l,m}} \omega_i v_{l,i}}{\sum_{i \in \mathcal{G}_{l,m}} \omega_i}
$$

Це дозволено **лише для semantic memory**.

---

## 8. Exact Recent Memory (ERM)

ERM потрібна для точного short-range recall.

Нехай recent window має розмір $W_r$.

$$
\mathcal{R}^{(t)}
=
\left\{
(\xi_j,\; \kappa_j,\; \tau_j)
\right\}_{j=1}^{W_r}
$$

де:

- $\xi_j$ — raw content slot:
  - token id
  - byte span
  - optional byte payload
- $\kappa_j$ — key
- $\tau_j$ — timestamp

ERM — це **ring buffer**. Жодної consolidation тут немає.

---

## 8.1 ERM write

Recent key:

$$
\kappa_t = W_{\mathrm{erm}}^{(\mathrm{write})} s_0^{(t)} + b_{\mathrm{erm}}^{(\mathrm{write})}
$$

Raw slot:

$$
\xi_t = (x_t,\; A_T(t),\; B[A_T(t)])
$$

Write pointer:

$$
p_{\mathrm{write}}^{(t+1)} = (p_{\mathrm{write}}^{(t)} + 1) \bmod W_r
$$

---

## 8.2 ERM query

Після semantic fusion будується:

$$
q_{\mathrm{erm}}^{(t)} = W_{\mathrm{erm}}^{(\mathrm{query})} h^{(t)} + b_{\mathrm{erm}}^{(\mathrm{query})}
$$

ERM attention:

$$
a_{\mathrm{erm},j}^{(t)}
=
\mathrm{softmax}_j
\left(
\frac{\langle q_{\mathrm{erm}}^{(t)}, \kappa_j\rangle}{\sqrt{d_{\mathrm{erm}}}}
\right)
$$

---

## 8.3 ERM copy distribution

Recent exact copy distribution over vocabulary:

$$
p_{\mathrm{erm}}(v \mid t)
=
\sum_{j=1}^{W_r}
a_{\mathrm{erm},j}^{(t)} \cdot \mathbf{1}[x_j = v]
$$

За потреби може існувати byte-copy distribution:

$$
p_{\mathrm{erm}}^{(\mathrm{byte})}(b \mid t)
=
\sum_{j=1}^{W_r}
a_{\mathrm{erm},j}^{(t)} \cdot \mathbf{1}[b \in B[A_T(j)]]
$$

---

## 9. Exact Episodic Memory (EEM)

EEM — це long-range exact memory.

Raw spans тут **immutable**.

Нехай chunk `m` визначається як:

$$
C_m =
\left(
B[s_m:e_m),
\;
T[u_m:v_m),
\;
\bar \kappa_m,
\;
\psi_m,
\;
t_m^{(\mathrm{start})},
\;
t_m^{(\mathrm{end})}
\right)
$$

де:

- $B[s_m:e_m)$ — raw UTF-8 bytes chunk
- $T[u_m:v_m)$ — token slice chunk
- $\bar \kappa_m$ — chunk key
- $\psi_m$ — metadata:
  - file id
  - symbol id
  - language id
  - chunk type
  - line range
  - scope range
  - rarity markers

---

## 9.1 Chunk creation policy

Chunk створюється:

- на function/class boundary;
- на block close;
- на file split boundary;
- на size threshold;
- на explicit important span event.

Chunk size:

$$
\ell_m^{(\mathrm{tok})} \in [L_{\min}^{(\mathrm{tok})}, L_{\max}^{(\mathrm{tok})}]
$$

$$
\ell_m^{(\mathrm{byte})} \in [L_{\min}^{(\mathrm{byte})}, L_{\max}^{(\mathrm{byte})}]
$$

---

## 9.2 Chunk key

Chunk summary state:

$$
\bar s_m = \mathrm{Pool}\left(\{s_0^{(i)}\}_{i=u_m}^{v_m-1}\right)
$$

Chunk key:

$$
\bar \kappa_m
=
W_{\mathrm{eem}}^{(\mathrm{write})} \bar s_m + b_{\mathrm{eem}}^{(\mathrm{write})}
$$

In-chunk pointer keys:

$$
\xi_{m,r}
=
W_{\mathrm{ptr}}^{(\mathrm{write})} s_0^{(u_m+r)} + b_{\mathrm{ptr}}^{(\mathrm{write})}
$$

---

## 9.3 EEM retrieval

Chunk query:

$$
q_{\mathrm{eem}}^{(t)} = W_{\mathrm{eem}}^{(\mathrm{query})} h^{(t)} + b_{\mathrm{eem}}^{(\mathrm{query})}
$$

Chunk score:

$$
\beta_m^{(t)}
=
\frac{\langle q_{\mathrm{eem}}^{(t)}, \bar \kappa_m \rangle}{\sqrt{d_{\mathrm{eem}}}}
+
\eta_{\mathrm{time}} g_{\mathrm{time}}(t-t_m)
+
\eta_{\mathrm{meta}} g_{\mathrm{meta}}(\psi_m, h^{(t)})
$$

Top chunks:

$$
\mathcal{C}_{\mathrm{eem}}^{(t)} = \mathrm{TopK}_m(\beta_m^{(t)})
$$

---

## 9.4 In-chunk pointering

Pointer query:

$$
q_{\mathrm{ptr}}^{(t)} = W_{\mathrm{ptr}}^{(\mathrm{query})} h^{(t)} + b_{\mathrm{ptr}}^{(\mathrm{query})}
$$

For token offset `r` inside chunk `m`:

$$
\pi_{m,r}^{(t)}
=
\frac{\langle q_{\mathrm{ptr}}^{(t)}, \xi_{m,r} \rangle}{\sqrt{d_{\mathrm{ptr}}}}
+
\eta_{\mathrm{loc}} g_{\mathrm{loc}}(r, \psi_m)
$$

Pointer distribution:

$$
\tilde \pi_{m,r}^{(t)}
=
\mathrm{softmax}_{m,r}(\pi_{m,r}^{(t)})
$$

Episodic copy distribution:

$$
p_{\mathrm{eem}}(v \mid t)
=
\sum_{m \in \mathcal{C}_{\mathrm{eem}}^{(t)}} \sum_{r}
\tilde \pi_{m,r}^{(t)} \cdot \mathbf{1}[x_{m,r}=v]
$$

---

## 10. Repository Graph Memory (RGM)

Програмування — це не просто послідовність токенів, а граф.

Нехай repository graph:

$$
\mathcal{G} = (\mathcal{V}, \mathcal{E})
$$

де вузли:

- files
- symbols
- imports
- functions
- classes
- tests
- diagnostics

Edges:

- defines
- calls
- imports
- overrides
- references
- tested-by
- fails-with

---

## 10.1 Node memory

Кожен graph node `q` має:

$$
g_q = (k_q^{(\mathrm{graph})}, v_q^{(\mathrm{graph})}, \psi_q^{(\mathrm{graph})})
$$

---

## 10.2 Graph query

$$
q_{\mathrm{graph}}^{(t)}
=
W_{\mathrm{graph}}^{(\mathrm{query})}
[h^{(t)};\; c_{\mathrm{scope}}^{(t)}]
+
b_{\mathrm{graph}}^{(\mathrm{query})}
$$

Graph node score:

$$
\zeta_q^{(t)}
=
\frac{\langle q_{\mathrm{graph}}^{(t)}, k_q^{(\mathrm{graph})} \rangle}{\sqrt{d_g}}
+
\eta_{\mathrm{samefile}} \mathbf{1}[q \in \mathrm{samefile}]
+
\eta_{\mathrm{import}} \mathbf{1}[q \in \mathrm{importclosure}]
+
\eta_{\mathrm{symbol}} \mathbf{1}[q \in \mathrm{currentsymbolclosure}]
$$

Retrieved graph context:

$$
o_{\mathrm{graph}}^{(t)}
=
\sum_{q \in \mathcal{C}_{\mathrm{graph}}^{(t)}}
\mathrm{softmax}_q(\zeta_q^{(t)}) \cdot v_q^{(\mathrm{graph})}
$$

---

## 11. Semantic fusion

Semantic memory outputs from all HSSM levels:

$$
o_l^{(t)}, \quad l=0,\dots,L
$$

Level gating:

$$
\gamma_l^{(t)}
=
\mathrm{softmax}_l
\left(
\frac{w_l^\top \mathrm{LN}(s_{\mathrm{master}}^{(t)})}{\sqrt{d_0}}
\right)
$$

Fused semantic context:

$$
c_{\mathrm{sem}}^{(t)}
=
\sum_{l=0}^{L}
\gamma_l^{(t)} \cdot W_l^{(\mathrm{out})} o_l^{(t)}
+
W_{\mathrm{graph}}^{(\mathrm{out})} o_{\mathrm{graph}}^{(t)}
$$

Base hidden state:

$$
h^{(t)}
=
\mathrm{FFN}\left(
\mathrm{LN}\left(
c_{\mathrm{sem}}^{(t)} + W_{\mathrm{skip}} s_0^{(t)}
\right)
\right)
$$

---

## 12. Retrieval Router

Модель не повинна викликати всі memory lanes на кожному токені без потреби.

Router input:

$$
u_{\mathrm{route}}^{(t)}
=
\left[
\mathrm{LN}(s_{\mathrm{master}}^{(t)});
h^{(t)};
\hat H_{\mathrm{lm}}^{(t)};
\hat H_{\mathrm{sem}}^{(t)};
\hat H_{\mathrm{erm}}^{(t)};
\hat H_{\mathrm{eem}}^{(t)};
\hat H_{\mathrm{graph}}^{(t)};
\phi^{(t)}
\right]
$$

де:

- $\hat H_\cdot$ — entropy / uncertainty features
- $\phi^{(t)}$ — cheap metadata features:
  - token class
  - rarity
  - string/number/comment flags
  - current syntax state

Router weights:

$$
\rho^{(t)}
=
\mathrm{softmax}\left(
W_{\mathrm{route}} u_{\mathrm{route}}^{(t)} + b_{\mathrm{route}}
\right)
$$

with components:

$$
\rho^{(t)}
=
\left[
\rho_{\mathrm{lm}}^{(t)},
\rho_{\mathrm{sem}}^{(t)},
\rho_{\mathrm{erm}}^{(t)},
\rho_{\mathrm{eem}}^{(t)},
\rho_{\mathrm{graph}}^{(t)}
\right]
$$

---

## 12.1 Compute gating

Під час inference lane `m` викликається тільки якщо:

$$
\rho_m^{(t)} > \theta_m
$$

або lane входить у `top-2` router weights.

Це забезпечує energy efficiency.

---

## 13. Output heads

## 13.1 Base LM head

$$
z_{\mathrm{lm}}^{(t)} = W_{\mathrm{vocab}} h^{(t)} + b_{\mathrm{vocab}}
$$

$$
p_{\mathrm{lm}}(v \mid t) = \mathrm{softmax}(z_{\mathrm{lm}}^{(t)})
$$

---

## 13.2 Semantic prior head

За потреби semantic retrieved context може створювати auxiliary distribution:

$$
z_{\mathrm{sem}}^{(t)} = W_{\mathrm{sem}} c_{\mathrm{sem}}^{(t)} + b_{\mathrm{sem}}
$$

$$
p_{\mathrm{sem}}(v \mid t) = \mathrm{softmax}(z_{\mathrm{sem}}^{(t)})
$$

---

## 13.3 Final distribution

Фінальна next-token distribution:

$$
p(v \mid t)
=
\rho_{\mathrm{lm}}^{(t)} p_{\mathrm{lm}}(v \mid t)
+
\rho_{\mathrm{sem}}^{(t)} p_{\mathrm{sem}}(v \mid t)
+
\rho_{\mathrm{erm}}^{(t)} p_{\mathrm{erm}}(v \mid t)
+
\rho_{\mathrm{eem}}^{(t)} p_{\mathrm{eem}}(v \mid t)
+
\rho_{\mathrm{graph}}^{(t)} p_{\mathrm{graph}}(v \mid t)
$$

де $p_{\mathrm{graph}}$ може бути:

- або copy-like distribution from graph literals/symbols,
- або graph-conditioned vocabulary projection.

---

## 14. Training objectives

## 14.1 Autoregressive loss

$$
\mathcal{L}_{\mathrm{AR}}
=
-\sum_{t=1}^{T-1}
\log p(x_{t+1} \mid x_{\le t})
$$

---

## 14.2 Infill loss

Для masked span $[a,b)$:

$$
\mathcal{L}_{\mathrm{INFILL}}
=
-\sum_{t=a}^{b-1}
\log p(x_t \mid x_{<a}, x_{\ge b})
$$

---

## 14.3 Hierarchical consistency

$$
\mathcal{L}_{\mathrm{Hier}}
=
\sum_{l=1}^{L}
\sum_{t:\, m_l^{(t)}=1}
\left\|
s_l^{(t)}
-
\mathrm{Proj}_{d_l}(\bar s_{l-1}^{(t)})
\right\|_2^2
$$

---

## 14.4 Sparse retrieval entropy

$$
\mathcal{L}_{\mathrm{Sparse}}
=
\sum_{l,t}
\mathrm{Entropy}(\alpha_l^{(t)})
$$

---

## 14.5 Recent copy supervision

Якщо target token присутній у ERM:

$$
\mathcal{L}_{\mathrm{CopyR}}
=
-\log p_{\mathrm{erm}}(x_{t+1}\mid t)
$$

---

## 14.6 Episodic pointer loss

Якщо target присутній у вибраному chunk:

$$
\mathcal{L}_{\mathrm{Ptr}}
=
-\log \tilde \pi_{m^\star,r^\star}^{(t)}
$$

---

## 14.7 Symbol linking loss

Для поточного token/state та істинного symbol node $q^\star$:

$$
\mathcal{L}_{\mathrm{Sym}}
=
-\log
\frac{\exp(\zeta_{q^\star}^{(t)})}
{\sum_{q \in \mathcal{N}(t)} \exp(\zeta_q^{(t)})}
$$

---

## 14.8 Routing loss

Для задач, де відомо, який lane бажаний:

$$
\mathcal{L}_{\mathrm{Route}}
=
-\sum_t \sum_m y_m^{(t)} \log \rho_m^{(t)}
$$

---

## 14.9 Energy-aware auxiliary penalty

Нехай per-step compute proxy:

$$
\hat C^{(t)}
=
c_{\mathrm{base}}
+
\sum_m \rho_m^{(t)} c_m
+
\chi_{\mathrm{maint}}^{(t)} c_{\mathrm{maint}}
$$

Тоді:

$$
\mathcal{L}_{\mathrm{Energy}}
=
\lambda_{\mathrm{energy}}
\sum_t \hat C^{(t)}
$$

Цей термін має бути слабким.

---

## 14.10 Diagnostic-only terms

TR magnitude:

$$
\mathcal{L}_{\mathrm{TRdiag}}
=
\sum \|G\|_F^2
$$

Ortho drift:

$$
\mathcal{L}_{\mathrm{Orthodiag}}
=
\sum \|G^\top G - I\|_F^2
$$

Ці terms можуть моніторитись і семплюватись, але не повинні руйнувати core objective.

---

## 14.11 Повна train objective

$$
\mathcal{L}
=
\lambda_{\mathrm{AR}} \mathcal{L}_{\mathrm{AR}}
+
\lambda_{\mathrm{INFILL}} \mathcal{L}_{\mathrm{INFILL}}
+
\lambda_{\mathrm{Hier}} \mathcal{L}_{\mathrm{Hier}}
+
\lambda_{\mathrm{Sparse}} \mathcal{L}_{\mathrm{Sparse}}
+
\lambda_{\mathrm{CopyR}} \mathcal{L}_{\mathrm{CopyR}}
+
\lambda_{\mathrm{Ptr}} \mathcal{L}_{\mathrm{Ptr}}
+
\lambda_{\mathrm{Sym}} \mathcal{L}_{\mathrm{Sym}}
+
\lambda_{\mathrm{Route}} \mathcal{L}_{\mathrm{Route}}
+
\lambda_{\mathrm{Energy}} \mathcal{L}_{\mathrm{Energy}}
$$

Recommended policy:

- optimize only gradient-useful terms;
- keep diagnostic terms outside the main objective by default.

---

## 14.12 Stable joint training of all components

The train-coupled system is split into five primary parameter blocks:

1. **Backbone / HSSM**: $\Theta_{\mathrm{B}}$
2. **Semantic memory (SHM query/key/value projections, hot read heads)**: $\Theta_{\mathrm{S}}$
3. **Exact Recent Memory adapters (ERM query/copy heads)**: $\Theta_{\mathrm{R}}$
4. **Exact Episodic Memory adapters (EEM chunk query/pointer heads)**: $\Theta_{\mathrm{E}}$
5. **Router + output heads**: $\Theta_{\mathrm{U}}$

Repository Graph Memory is treated differently:

- the **graph store itself** is mostly non-gradient structured data;
- only graph query projections, node adapters, and graph-aware heads are optimized online;
- therefore, graph-trainable parameters are attached to $\Theta_{\mathrm{U}}$ unless a dedicated graph encoder is introduced.

This distinction is mandatory. Otherwise the model becomes unstable by trying to backpropagate through mutable indexes, immutable byte archives, and maintenance operations that are not meant to be in the main autograd graph.

---

## 14.12.1 Loss normalization and activation masks

Every auxiliary loss must be normalized by the number of valid supervision events:

$$
\bar{\mathcal{L}}_i
=
\frac{\mathcal{L}_i}{N_i + \varepsilon}
$$

where:

- $i$ indexes a loss term;
- $N_i$ is the number of valid events for that term in the batch;
- $\varepsilon > 0$ is a small constant.

Examples:

- for $\mathcal{L}_{\mathrm{CopyR}}$, only steps where the target token is present in ERM count toward $N_i$;
- for $\mathcal{L}_{\mathrm{Ptr}}$, only steps where the target is inside a retrieved EEM chunk count toward $N_i$;
- for $\mathcal{L}_{\mathrm{Sym}}$, only steps with valid symbol labels count toward $N_i$.

Each trainable term is also guarded by a phase activation gate:

$$
a_i(u) \in [0,1]
$$

where $u$ is the global optimizer step. The effective objective is:

$$
\mathcal{L}_{\mathrm{train}}(u)
=
\sum_i
a_i(u)\,
\lambda_i^{\mathrm{eff}}(u)\,
\bar{\mathcal{L}}_i(u)
$$

Diagnostic-only terms remain outside $\mathcal{L}_{\mathrm{train}}$.

---

## 14.12.2 Dynamic loss balancing

Auxiliary losses must not silently dominate autoregressive learning. Therefore each non-core term uses a capped effective weight:

$$
\lambda_i^{\mathrm{eff}}(u)
=
\lambda_i
\min\left(
1,\;
r_i \cdot
\frac{\mathrm{EMA}_W[\bar{\mathcal{L}}_{\mathrm{AR}}]}
{\mathrm{EMA}_W[\bar{\mathcal{L}}_i] + \varepsilon}
\right)
$$

where:

- $\mathrm{EMA}_W[\cdot]$ is a moving average over a window $W$;
- $r_i$ is the maximum allowed ratio of auxiliary contribution to AR contribution.

Recommended initial caps:

- $r_{\mathrm{Hier}} = 0.25$
- $r_{\mathrm{Sparse}} = 0.10$
- $r_{\mathrm{CopyR}} = 0.50$
- $r_{\mathrm{Ptr}} = 0.50$
- $r_{\mathrm{Sym}} = 0.25$
- $r_{\mathrm{Route}} = 0.10$
- $r_{\mathrm{Energy}} = 0.05$

Interpretation:

- hierarchy and routing may regularize;
- copy and pointer losses may become strong once their lanes are proven useful;
- energy penalty must remain weak.

If an auxiliary term still spikes due to noise, the phase gate $a_i(u)$ must be held below $1$ until the corresponding lane is stable.

---

## 14.12.3 Gradient-flow rules

Stable training depends on explicit gradient boundaries.

### Gradient is allowed through:

- token/byte/structure encoders for the current batch;
- HSSM updates within the current truncated unroll;
- SHM query/key/value projections for hot memory written and read inside the same train step;
- ERM query/copy heads;
- EEM query and in-chunk pointer heads;
- router and final output heads.

### Gradient is not allowed through:

- raw UTF-8 byte storage itself;
- token IDs and alignment maps as discrete metadata;
- immutable EEM chunk payloads after they are committed;
- search-tree construction;
- consolidation clustering;
- graph indexing and repository scan passes;
- memory state carried across independent training sequences.

Formally, for any persistent state $M^{(u)}$ carried from optimizer step $u$ to $u+1$:

$$
\frac{\partial \mathcal{L}^{(u+1)}}{\partial M^{(u)}} = 0
$$

unless that state is still inside the same truncated training unroll.

If truncated BPTT length is $K_{\mathrm{bptt}}$, then:

$$
\frac{\partial \mathcal{L}_t}{\partial s_{t-k}} = 0
\quad \text{for } k > K_{\mathrm{bptt}}
$$

after the detach boundary.

This rule is required for both stability and energy efficiency.

---

## 14.12.4 Router warmup and anti-collapse protocol

The router must not learn from scratch in full free-routing mode. It should first be trained with soft oracle guidance.

Let $A_m^{(t)} \in \{0,1\}$ indicate whether lane $m$ can recover the target at step $t$:

- ERM lane: target exists in recent exact window;
- EEM lane: target exists in selected chunk;
- graph lane: correct symbol node is available;
- semantic lane: always available as fallback.

Construct a soft oracle target:

$$
y_m^{(t)}
=
\frac{\exp(\nu A_m^{(t)} + b_m)}
{\sum_j \exp(\nu A_j^{(t)} + b_j)}
$$

where $\nu$ controls oracle sharpness and $b_m$ encodes optional priors.

During warmup, the effective router used in the mixture is:

$$
\tilde{\rho}_m^{(t)}
=
(1-\beta_u)\, y_m^{(t)}
+
\beta_u\, \rho_m^{(t)}
$$

with $\beta_u$ ramped from $0$ to $1$ over router warmup.

Hard rule:

- before all memory lanes are individually validated, use $\tilde{\rho}$ for mixture;
- only after warmup completion switch fully to learned $\rho$.

To avoid early router collapse:

- enforce top-2 routing, not top-1, during warmup;
- apply lane dropout to the dominant lane with small probability;
- keep a minimum entropy floor for router logits in early phases.

One practical entropy floor is:

$$
\mathcal{L}_{\mathrm{EntFloor}}
=
\lambda_{\mathrm{ent}}
\sum_t
\max(0, H_{\min} - H(\rho^{(t)}))
$$

This term is phase-local and should be disabled once routing has diversified.

---

## 14.12.5 Phase curriculum

The model must not turn on all subsystems at once.

### Phase 0: Representation and indexing bootstrap

- train or fit the code-aware tokenizer;
- build token-byte alignment maps;
- build initial repo graph indexes;
- verify chunking and structure-boundary extraction;
- no end-to-end model training yet.

### Phase A: Stable semantic backbone

Train:

- $\Theta_{\mathrm{B}}$
- $\Theta_{\mathrm{S}}$
- base LM head inside $\Theta_{\mathrm{U}}$

Active losses:

- $\mathcal{L}_{\mathrm{AR}}$
- $\mathcal{L}_{\mathrm{INFILL}}$
- $\mathcal{L}_{\mathrm{Hier}}$
- $\mathcal{L}_{\mathrm{Sparse}}$

Disabled:

- ERM copy loss
- EEM pointer loss
- graph symbol loss
- free router mixing

Policy:

- router is forced to semantic lane only;
- cold consolidation is delayed or very infrequent;
- diagnostics are sampled, not dense.

Exit criteria:

- no NaNs for a long validation window;
- AR improves monotonically in EMA terms;
- hierarchy loss plateaus below a configured threshold;
- hot/cold semantic reads do not explode latency.

### Phase B: ERM introduction

Train:

- $\Theta_{\mathrm{B}}$
- $\Theta_{\mathrm{S}}$
- $\Theta_{\mathrm{R}}$
- copy-aware part of $\Theta_{\mathrm{U}}$

New active loss:

- $\mathcal{L}_{\mathrm{CopyR}}$

Policy:

- router may choose between semantic and recent-exact lanes only;
- use oracle-guided $\tilde{\rho}$;
- ERM is always written every step;
- EEM and graph lane remain frozen.

Exit criteria:

- recent-copy accuracy clears a target threshold on synthetic and real probes;
- router uses ERM when exact target is present, but does not overuse it when absent.

### Phase C: EEM introduction

Train:

- $\Theta_{\mathrm{E}}$
- pointer-aware part of $\Theta_{\mathrm{U}}$
- all previous active blocks remain trainable

New active loss:

- $\mathcal{L}_{\mathrm{Ptr}}$

Policy:

- EEM is built from immutable chunks only;
- chunk selection may be teacher-guided early on;
- router now mixes semantic + ERM + EEM;
- graph lane still frozen.

Exit criteria:

- chunk retrieval recall is stable;
- pointer accuracy is materially above random and above semantic-only baseline;
- EEM does not destroy tokens/sec beyond the accepted budget.

### Phase D: Graph lane introduction

Train:

- graph query projections and graph-aware heads in $\Theta_{\mathrm{U}}$
- optional graph-node adapters if used

New active loss:

- $\mathcal{L}_{\mathrm{Sym}}$

Policy:

- repository graph payload stays external or semi-static;
- graph refresh happens offline or at low frequency;
- graph lane is enabled only on tasks that actually require symbol reasoning.

Exit criteria:

- symbol-link top-1/top-k reaches target levels;
- graph lane improves cross-file reasoning without collapsing router diversity.

### Phase E: Full router activation

Train:

- full $\Theta_{\mathrm{U}}$
- all already enabled memory lanes

New active loss:

- full $\mathcal{L}_{\mathrm{Route}}$
- optional weak $\mathcal{L}_{\mathrm{Energy}}$

Policy:

- switch from oracle-guided $\tilde{\rho}$ to learned $\rho$ gradually;
- top-2 routing may remain as a permanent efficiency constraint;
- per-lane usage is monitored continuously.

Exit criteria:

- no persistent single-lane collapse;
- validation quality improves over phase-isolated baselines;
- compute overhead stays inside budget.

### Phase F: Joint fine-tuning

Train all enabled blocks jointly with full task mix.

Policy:

- auxiliary caps remain active;
- maintenance is scheduled, never dense;
- exact-recall, repo-edit, and long-context benchmarks all gate acceptance.

---

## 14.12.6 Optimizer groups and learning-rate ratios

Use AdamW or a similarly stable optimizer with parameter groups:

$$
\eta_{\mathrm{B}} = \eta
$$

$$
\eta_{\mathrm{S}} = 0.7 \eta
$$

$$
\eta_{\mathrm{R}} = 0.7 \eta
$$

$$
\eta_{\mathrm{E}} = 0.5 \eta
$$

$$
\eta_{\mathrm{U}} =
\begin{cases}
0.3 \eta, & \text{during router warmup} \\
0.7 \eta, & \text{after router stabilizes}
\end{cases}
$$

Recommended gradient clipping by group:

- backbone/HSSM: $\|g\|_2 \le 1.0$
- SHM: $\|g\|_2 \le 0.5$
- ERM: $\|g\|_2 \le 0.5$
- EEM: $\|g\|_2 \le 0.5$
- router/heads: $\|g\|_2 \le 0.25$

Reason:

- router and pointer heads are high-variance;
- memory projections are more fragile than the backbone;
- clipping per group prevents one unstable lane from corrupting the whole step.

---

## 14.12.7 Maintenance scheduling

Memory maintenance must be separated from the main gradient path.

Define the consolidation trigger:

$$
\chi_{\mathrm{cons}}(u)
=
\mathbf{1}\Big[
u > U_{\mathrm{warm}}
\;\land\;
\mathrm{occ}_{\mathrm{hot}}(u) > \tau_{\mathrm{occ}}
\;\land\;
u \bmod K_{\mathrm{cons}} = 0
\;\land\;
\Delta \mathrm{EMA}_W[\mathcal{L}_{\mathrm{AR}}] \le \delta_{\mathrm{spike}}
\Big]
$$

Interpretation:

- do not consolidate too early;
- only consolidate if hot memory occupancy justifies it;
- do not consolidate during a loss spike.

Maintenance rules:

- SHM hot write: every step
- ERM ring write: every step
- EEM chunk finalization: only on chunk boundary or structural close
- SHM consolidation: only when $\chi_{\mathrm{cons}}(u)=1$
- search-tree rebuild: only after a consolidation or bulk cold-memory update
- graph refresh: offline, per repo snapshot, or at low periodicity

All maintenance operations run under `no_grad()` unless there is a dedicated auxiliary optimization pass designed specifically for them.

---

## 14.12.8 Task-mixture curriculum

The training data mixture must also follow the phase schedule.

Recommended starting curriculum:

- **Phase A**
  - 70% autoregressive code modeling
  - 30% infill / span completion

- **Phase B**
  - 55% autoregressive code modeling
  - 20% infill
  - 25% recent exact-recall and copy probes

- **Phase C**
  - 45% autoregressive code modeling
  - 15% infill
  - 20% recent copy probes
  - 20% episodic chunk-recall tasks

- **Phase D / E / F**
  - 35% autoregressive code modeling
  - 15% infill
  - 15% recent copy probes
  - 15% episodic recall
  - 20% repo-level tasks (symbol linking, definition-use matching, edit/fix tasks)

This mixture is a default, not a law. But the key constraint is fixed:

- do not train the router or graph lane mostly on plain next-token tasks;
- explicitly show the model tasks that require exact recall and repo reasoning.

---

## 14.12.9 Per-step training algorithm

For each optimizer step:

1. sample a batch with task labels and phase-appropriate supervision;
2. build or load alignment metadata and structure boundaries;
3. run forward pass only through lanes enabled in the current phase;
4. compute oracle availability masks $A_m^{(t)}$ for enabled lanes;
5. compute router weights $\rho$ and warmup mixture $\tilde{\rho}$ if router warmup is active;
6. compute all active normalized losses $\bar{\mathcal{L}}_i$;
7. compute effective weights $\lambda_i^{\mathrm{eff}}(u)$;
8. backpropagate only through $\mathcal{L}_{\mathrm{train}}(u)$;
9. apply per-group gradient clipping;
10. take optimizer step;
11. if $\chi_{\mathrm{cons}}(u)=1$, run maintenance under `no_grad()`;
12. log:
   - AR
   - active auxiliary losses
   - router entropy
   - per-lane utilization
   - exact recall metrics
   - tokens/sec
   - maintenance latency

This order is part of the concept, not an implementation detail.

---

## 14.12.10 Failure detectors and mandatory reactions

The concept requires explicit failure handling.

### Router collapse

Condition:

- one lane gets $> 95\%$ mean routing mass over a long window before all lanes are validated.

Reaction:

- reduce $\beta_u$ growth;
- re-enable oracle-guided mixture;
- increase lane dropout on the dominant lane;
- increase router entropy floor temporarily.

### Auxiliary domination

Condition:

- any auxiliary contribution persistently exceeds its configured cap relative to AR.

Reaction:

- reduce $\lambda_i^{\mathrm{eff}}$ automatically;
- if needed, freeze that lane for a short recovery interval.

### Memory-maintenance instability

Condition:

- AR jumps immediately after consolidation windows;
- tokens/sec collapses after maintenance is enabled.

Reaction:

- increase $K_{\mathrm{cons}}$;
- increase occupancy threshold $\tau_{\mathrm{occ}}$;
- disable consolidation during the next recovery window;
- keep cold compression and tree build in FP32 if numerically fragile.

### Exact-lane hallucination

Condition:

- ERM/EEM lane receives high router weight even when target is absent or recall accuracy is low.

Reaction:

- strengthen router supervision;
- lower lane prior;
- require availability mask consistency during warmup.

### Graph overreach

Condition:

- graph lane dominates on tasks that are local-file or recent-context only.

Reaction:

- lower graph prior outside repo tasks;
- gate graph lane by task type;
- penalize unnecessary graph usage through the energy term.

---

## 14.12.11 Phase-transition metrics

A phase transition is allowed only if all required metrics pass.

Recommended minimum checks:

- **Phase A -> B**
  - stable AR improvement
  - no gradient explosions
  - semantic-only baseline established

- **Phase B -> C**
  - recent-copy top-1 accuracy clears target on held-out probes
  - router uses ERM selectively, not blindly

- **Phase C -> D**
  - episodic chunk recall and pointer accuracy materially exceed semantic-only baseline

- **Phase D -> E**
  - symbol-link metrics improve on cross-file tasks
  - graph lane latency remains acceptable

- **Phase E -> F**
  - router diversity is healthy
  - full mixture outperforms ablations
  - energy and throughput remain within budget

If a phase fails its exit checks, training must not progress just because loss is still decreasing.

---

## 15. Inference algorithms

## 15.1 Streaming completion

For each token step:

1. encode current token into code-token + byte + structure features
2. update HSSM levels by hybrid schedule
3. write active levels into SHM hot memory
4. update ERM ring
5. if chunk boundary reached, finalize an EEM chunk
6. retrieve:
   - hot semantic
   - cold semantic if routed
   - recent exact if routed
   - episodic exact if routed
   - graph memory if routed
7. fuse context
8. produce mixture distribution
9. sample / argmax next token

---

## 15.2 Edit / refactor mode

In edit mode inference runs over:

- file-local context
- changed span context
- retrieved symbol definitions
- tests/diagnostics graph

Output is not only raw next-token generation, but also optional:

- span replacement
- edit diff
- AST-consistent patch

---

## 15.3 Repo question answering mode

For repo tasks:

- graph lane and episodic lane receive higher router prior;
- recent lane still handles exact quoted fragments;
- semantic lane handles reasoning and summarization.

---

## 16. Efficiency and energy policy

## 16.1 Hot path

Always cheap:

- token embedding
- HSSM update
- hot semantic read
- ERM ring write/read

Conditional:

- cold semantic tree read
- episodic retrieval
- graph retrieval

Deferred:

- consolidation
- tree rebuild
- chunk reindexing
- diagnostic regularizer evaluation

---

## 16.2 Sampled diagnostics

Expensive diagnostics run every $K$ steps:

$$
\delta_{\mathrm{diag}}^{(t)} = \mathbf{1}[t \equiv 0 \pmod K]
$$

---

## 16.3 No decompress-on-read policy

Compressed banks may exist for storage, but read-path should prefer dense candidate cache whenever possible.

This is mandatory for efficiency.

---

## 16.4 Mixed precision policy

- standard matmul-heavy hot path may use AMP;
- numerically fragile operations must force float32:
  - SVD
  - tree build math if unstable
  - certain normalization/orthogonality diagnostics

---

## 16.5 Complexity targets

Let:

- $W_r$ — recent exact window
- $K_e$ — number of episodic chunks retrieved
- $\bar \ell$ — mean chunk length
- $b$ — semantic beam width

Then per-token target complexity is:

### HSSM:

$$
O\left(\sum_{l=0}^{L} d_l^2\right)
$$

### Semantic read:

$$
O\left(
\sum_{l=0}^{L}
(\log N_l + b d_l)
\right)
$$

### ERM:

$$
O(W_r d)
$$

### EEM:

$$
O(\log N_{\mathrm{chunk}} + K_e \bar \ell d)
$$

### Graph:

$$
O(\log |\mathcal{V}| + K_g d)
$$

Total target hot-path:

$$
O\left(
\sum_l d_l^2
+
\sum_l \log N_l
+
W_r d
+
K_e \bar \ell d
+
K_g d
\right)
$$

All heavy maintenance must stay off the hot path whenever possible.

---

## 17. Hyperparameter ranges

Recommended initial ranges:

| Parameter | Recommended range |
|-----------|-------------------|
| $L$ | 5–6 |
| $d_0$ | 512–2048 |
| $k$ | 2 |
| $\alpha$ | 1.5–2.0 |
| hot semantic slots | 32–256 per level |
| cold semantic slots | 128–4096 per level |
| $W_r$ | 128–512 tokens |
| episodic chunk size | 32–256 tokens |
| episodic top-$K$ | 1–8 chunks |
| semantic beam width $b$ | 8–64 |
| graph top-$K$ | 4–32 |
| diagnostics period $K$ | 8–64 steps |

---

## 18. Invariants

These are hard architectural invariants.

1. **ERM raw content is never merged.**
2. **EEM raw spans are immutable.**
3. **Only semantic memory may be lossy-compressed.**
4. **Every token must retain byte alignment.**
5. **Every symbol-aware operation must preserve file/scope identity.**
6. **Diagnostic losses must not dominate training objective by accident.**
7. **Hot-path inference must not depend on global history scan.**

---

## 19. Benchmark and acceptance criteria

The model is only successful if it wins on at least one real axis against a strong code Transformer baseline.

Mandatory benchmark families:

1. **AR quality**
   - val loss / ppl on code corpora
2. **Exact recall**
   - identifier recall
   - string recall
   - number recall
   - quoted span recall
   - exact copy from 100 / 1k / 10k tokens back
3. **Repo reasoning**
   - symbol resolution
   - cross-file edit success
   - test-to-fix success
   - import/call-chain reasoning
4. **Efficiency**
   - tokens/sec
   - peak VRAM
   - maintenance overhead
5. **Long-context stability**
   - max stable context
   - degradation threshold vs baseline context
6. **Energy proxies**
   - wall-time per token
   - maintenance calls per 1k tokens
   - memory-retrieval count per token

Acceptance rule:

If the architecture is more complex and slower than a Transformer but not better on any important axis, the design is incomplete.

---

## 20. Recommended implementation order

### Phase A — stable semantic HTM

- stabilize HSSM
- stabilize SHM/TRAM
- sample diagnostics
- cheap hot/cold read path

### Phase B — ERM

- exact recent ring buffer
- copy distribution
- recent-copy loss

### Phase C — EEM

- immutable chunk store
- chunk retrieval
- pointer loss

### Phase D — RGM

- symbol graph
- import/call/test graph retrieval

### Phase E — Router

- mixture routing
- energy-aware gating

### Phase F — Full benchmark harness

- exact recall suite
- repo editing suite
- long-context sweep
- throughput/VRAM/energy proxies

---

## 21. Reference module breakdown

Recommended codebase layout:

- `tokenizer/`
  - lexer-aware tokenizer
  - byte fallback
  - alignment builder
- `encoders/`
  - token encoder
  - byte span encoder
  - structure encoder
- `hssm/`
  - hierarchical state updates
  - structural boundary scheduler
- `memory/semantic/`
  - hot/cold memory
  - consolidation
  - tree index
- `memory/exact_recent/`
  - ring buffer
  - copy head
- `memory/exact_episodic/`
  - chunk store
  - pointer index
- `memory/repo_graph/`
  - symbol graph
  - graph retrieval
- `router/`
  - lane gating
  - budget gating
- `losses/`
  - AR
  - infill
  - copy
  - pointer
  - route
  - energy
- `benchmarks/`
  - quality
  - exact recall
  - repo tasks
  - throughput
  - long context

---

## 22. Final summary

Фінальна модель для програмування повинна:

- мислити як **ієрархічна semantic model**
- пам’ятати як **lossless exact archive**
- орієнтуватись у коді як **repo-native graph model**
- генерувати як **mixture of LM + copy + pointer**
- працювати як **bounded-compute incremental system**

Коротко:

$$
\text{Final HTM for code}
=
\text{semantic hierarchy}
+
\text{exact local memory}
+
\text{exact archival memory}
+
\text{repo graph retrieval}
+
\text{energy-aware routing}
$$

Саме це і є повна фінальна консолідована концепція.
