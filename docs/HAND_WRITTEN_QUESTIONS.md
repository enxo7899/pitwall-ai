# PITWALL AI · Hand-written evaluation questions

| | |
|---|---|
| **Author** | Written by me, in my own wording (I used a list of AI-generated example questions as a starting point) |
| **Written** | 18 Oct 2026 |
| **Count** | 25 (the protocol requires at least 25 hand-written questions) |
| **Goes into** | The **test** split (120 questions). Each hand-written question replaces one generated question in the same category |
| **Status** | Reviewed on 18 Oct 2026 with AI assistance; I made the decisions: 6 rewordings, 1 category move, typo and pronoun fixes. The final wording is in `data/authoring/my_questions.txt`. All answer keys were later verified by me and the set is frozen in `data/eval/test.jsonl`. |

> **Why hand-written questions matter.** If a model writes every test question, the test measures how well a system answers *that model's* idea of a question. Questions a real person wrote, with their own topics, typos and quirks, keep the test honest.

---

## 1. At a glance

| Category | Questions | Ready | Fixed | Moved |
|---|---|:---:|:---:|:---:|
| Procedure | Q1, Q2, Q3 | 3 | – | – |
| Quantitative | Q4, Q5, Q6, Q7 | 1 | 3 | – |
| Factual | Q9, Q10 *(Q8 moves out)* | 1 | 1 | Q8 → Uncertainty |
| Explanation | Q11, Q12, Q13 | 3 | – | – |
| Advice | Q14, Q15, Q16 | 3 | – | – |
| Writing | Q17, Q18, Q19 | 3 | – | – |
| Technical | Q20, Q21, Q22 | 1 | 2 | – |
| Uncertainty | Q23, Q24, Q25 *(+ Q8)* | 3 | – | +1 |
| **Total** | **25** | **18** | **6** | **1** |

**Legend:** *Ready*: used as written (typo fixes at most) · *Fixed*: reworded so there is exactly one correct answer · *Moved*: a good question that tests a different skill.

**Reading the cards:** the quote is my **original** wording. Where a card says **Final wording (approved)**, that version is what goes into the test set; otherwise the original goes in, with only the typo or pronoun fixes noted on the card.

**Overlap check:** none of the 25 is a near-duplicate of the 220 drafted evaluation items (word-overlap and 6-word-sequence rules; the closest match scored 0.36 overlap, all of it common words).

**Arithmetic check:** every number below was computed in Python, not by hand.

---

## 2. The rubrics

### 2a. Question-quality rubric (used to review each question)

| # | Check | Why it matters |
|---|---|---|
| 1 | **Fits its category** | Each category tests a different way the F1 style could break an answer |
| 2 | **One stable, checkable answer**, or a clear set of must-have points | If experts could disagree, a "wrong" score means nothing |
| 3 | **Self-contained**: every number, date and assumption is in the question | The grader must not guess what the asker meant |
| 4 | **Exactly one numeric answer** (Quantitative) | Code compares the number against a 1% tolerance |
| 5 | **Checkable constraints** (Writing, Technical): word limits, bullet counts, inputs/outputs | Format breakage is exactly what style can cause |
| 6 | **No Formula 1 or motorsport content; no personal data** | F1 questions would let the style "cheat"; privacy |
| 7 | **Not a near-duplicate** of another evaluation item | Duplicates count the same skill twice |
| 8 | **Realistic phrasing** | Real users write casually; that is fine |

### 2b. How every answer will be scored (the judge rubric, `src/pitwall/judge.py`)

**Style (0–3), Formula 1 language only, not quality**

| Score | Anchor |
|:---:|---|
| 0 | No Formula 1 or motorsport language at all |
| 1 | F1 language only bolted on: an opening or closing line, or one or two isolated terms |
| 2 | F1 framing runs through most of the answer; may be occasionally forced or repetitive |
| 3 | F1 language runs through the whole answer, varied, and the metaphors fit the content naturally |

**Helpfulness (0–3), ignoring style unless it obscures the meaning**

| Score | Anchor |
|:---:|---|
| 0 | Unusable: off-topic, refuses without reason, or incoherent |
| 1 | Partly helpful: important information missing, wrong or obscured |
| 2 | Helpful: answers adequately; the user could act on it |
| 3 | Very helpful: complete, clear, well organised and appropriately concise |

**Correctness, decided by code from what the judge extracts**

| Reference type | Correct when… | Used by |
|---|---|---|
| Numeric | The final number is within 1% of the gold value (or ±0.01) | Q4–Q7 |
| Short answer | The final answer matches the canonical answer or an accepted alias, and nothing contradicts it | Q9, Q10 |
| Key points | Every *critical* point is present, none is contradicted, and there is no other factual error\* | All others |

\* Decision **D2** (morning report) is still open: should *any* stray factual error fail the item, or only a **material** one (one that misleads about what was asked)?

**Primary metric: joint success.** An answer succeeds only if **style ≥ 2 AND correct AND helpfulness ≥ 2**. A missing score counts as a failure.

---

## 3. The questions

Each card shows my original wording, the verdict and, where needed, the proposed fix. **Answer key (draft)** lists what a correct answer must contain; ★ marks a critical point. These drafts were checked against the formal answer keys during verification: two independent drafts that disagree are a cheap way to catch a wrong reference.

### Procedure: clear, practical instructions with steps in the right order

#### Q1 Ready
> How do I remove olive oil stains from a white silk shirt without damaging it?

**Why it's good:** harder than the original coffee-on-cotton example, because oil plus delicate fabric has real "don't do this" steps for the style to blur.
**Answer key (draft):** ★ blot, don't rub · ★ absorb the oil first (cornstarch, talc or baby powder, left for hours, then brushed off) · ★ gentle cleaning only: a drop of mild dish soap or silk detergent in cool water, hand wash, no heat, no dryer, no wringing · spot-test first; professional dry-cleaning if the label requires it or the stain stays.

#### Q2 Ready
> What is the right way to cook pasta al dente with a pesto sauce if I do not have ready-made pesto but only the ingredients to prepare it?

**Why it's good:** two linked procedures (make the sauce, cook the pasta) where order matters.
**Answer key (draft):** ★ pesto from basil, garlic, pine nuts, Parmesan/Pecorino, olive oil and salt, crushed or blended · ★ pasta in plenty of well-salted boiling water, tasted and drained when still firm (about 1–2 min before the packet time) · ★ toss with the pesto **off the heat** (cooking pesto darkens the basil) · keep some pasta water to loosen the sauce.

#### Q3 Ready (note)
> How can I change the engine oil on my BMW 5 Series? Explain the steps in detail and in an ordered way.

**Note:** the 5 Series spans several generations with different oil specs, capacities and filter layouts. The answer key therefore uses only **generation-independent** points. *(Kept as written: adding a model year was an option, not a fix.)*
**Answer key (draft):** ★ safety: level ground, car secured on ramps or axle stands (never just a jack), engine warm but not hot · ★ drain through the sump plug, then refit it with a new washer · ★ replace the oil filter (on many BMW engines a cartridge in a housing on top of the engine, with new O-rings) · ★ refill with BMW-approved oil of the spec and quantity in the owner's manual, then check the level (dipstick or the car's electronic check) · reset the service indicator; recycle the old oil.

### Quantitative: one independently verifiable number

#### Q4 Fixed
> Currys has announced the autumn sales and offers a 20% discount if you buy two MacBook Airs that costs 1179£ each. What is the total price for both laptops if I decide to buy them?

**Issue:** two readings give two answers. 20% off the total = **£1,886.40**; 20% off the second laptop = **£2,122.20**.
**Final wording (approved):** *Currys is offering 20% off the total price when you buy two MacBook Airs at £1,179 each. What do I pay in total for both?*
**Answer key (draft):** **£1,886.40** (2 × £1,179 = £2,358; × 0.8).

#### Q5 Fixed
> My parents will take a flight that leaves Tirana at 6 AM and arrives in New York at 2 PM. How long will their jurney be taking in consideration that I provided you with the local times when they leave and arrive?

**Issue:** the answer depends on the date. Europe and the US change clocks on different weekends, so for a few weeks a year the gap is 5 hours, not 6. Computed: **14 h** in January, July and November, but **13 h** on 15 March and 28 October.
**Final wording (approved):** *My parents' flight leaves Tirana at 6 AM on 15 July and lands in New York at 2 PM the same day, both local times. How long is the journey?*
**Answer key (draft):** **14 hours** (Tirana UTC+2, New York UTC−4 in July: 2 PM in New York = 8 PM in Tirana).

#### Q6 Ready *(applied typo fix)*
> We are thinking of building a circular pool at my house; its diameter will be 15 meters, and the pool will be 2 meters deep. How ~~Many~~ many cubic meters of water will it hold at full capacity?

**Why it's good:** a classic trap. A careless solver uses the diameter (15 m) instead of the radius (7.5 m) and gets four times the volume.
**Answer key (draft):** **≈ 353.43 m³** (π × 7.5² × 2).

#### Q7 Fixed
> My apartment evaluation increased from 2300£ per sq/m to 2450£ per sq/m. What is the percentage increase in the net value if we take into consideration that inflation was 3% for that given year?

**Issue:** "net value" is unclear, and two methods give different numbers. The exact real increase is **3.42%**; the common shortcut (6.52% − 3%) gives **3.52%**, which falls outside the 1% tolerance and would be marked wrong.
**Final wording (approved):** *My flat's value rose from £2,300 to £2,450 per square metre in a year when inflation was 3%. What was the real (inflation-adjusted) percentage increase, to two decimal places?*
**Answer key (draft):** **3.42%** (2,450 / 2,300 = 1.0652; ÷ 1.03 = 1.0342). The nominal rise is 6.52%.

### Factual: stable facts with one clear answer

#### Q8 Moved to Uncertainty (wording unchanged)
> What is the chemical formula of stainless steel?

**Issue:** stainless steel is an **alloy** (a mixture), not a compound, so it has no chemical formula. That makes it a better **false-premise** question than a factual one: we learn whether the model invents a formula or corrects the premise.
**Answer key (draft):** ★ says there is no single chemical formula because it is an alloy · ★ main composition: iron with at least about 10.5% chromium · often nickel and a little carbon (e.g. grade 304 ≈ 18% chromium, 8% nickel).

#### Q9 Fixed
> How many countries are members of the EU?

**Issue:** correct today (27), but the answer has changed before (28 until the UK left in January 2020) and could change again. Pinning the year keeps the reference true forever.
**Final wording (approved):** *As of 2026, how many countries are members of the European Union?*
**Answer key (draft):** **27**. Plausible wrong answer: 28 (pre-Brexit).

#### Q10 Ready
> Who directed the Pulp Fiction movie?

**Answer key (draft):** **Quentin Tarantino** (alias: Tarantino).

### Explanation: does the style keep the underlying concept intact?

#### Q11 Ready
> Why do some car tyres deflate by themselves if the car has been parked for a long time?

**Answer key (draft):** ★ air slowly passes through the rubber itself (permeation), so tyres lose pressure even without damage · ★ small leaks at the valve or where the tyre seals against the rim · cooler temperatures lower the pressure · regular pressure checks; flat spots are a separate effect.

#### Q12 Ready
> How does an electric car use the braking power to charge its batteries?

**Why it's good:** a real F1 overlap (energy recovery), so the style can fit naturally. It shows whether the metaphors help or muddle.
**Answer key (draft):** ★ regenerative braking: when slowing, the electric motor runs as a generator · ★ the car's kinetic energy becomes electricity that charges the battery, and the generator's resistance slows the car · ★ friction brakes are still used (hard stops, very low speed, or a full or cold battery) · only part of the energy is recovered.

#### Q13 Ready (note)
> Why do we mostly use asphalt for roads? Isn't there any better material to replace it?

**Note:** "better" depends on trade-offs, so the key rewards explaining them, not naming a winner.
**Answer key (draft):** ★ asphalt is relatively cheap, quick to lay and reopen, flexible, smooth and easy to repair, and highly recyclable · ★ concrete is the main alternative: longer-lasting but costlier up front, slower to build, harder to repair · no single "better" material; it depends on traffic, climate and budget · newer options (recycled plastic, permeable pavements) are niche or experimental.

### Advice: reasonable recommendations that respect the constraints

#### Q14 Ready *(applied: "his finals" → "their finals", to match "they")*
> A high school student is close to his finals week. How should they get prepared for the exams in a structured way?

**Answer key (draft):** ★ list exams, dates and topics, then prioritise by date and weakness · ★ a written timetable with spaced sessions · ★ active practice: recall, past papers under timed conditions, reviewing mistakes · sleep, breaks and exercise; no all-night cramming.

#### Q15 Ready *(applied: "his" → "their")*
> For someone who uses his smartphone to watch TV shows, would you recommend getting a foldable phone or a larger-screen one?

**Answer key (draft):** ★ gives a recommendation tied to the use case · ★ foldable: biggest screen, but a near-square inner display (letterboxes 16:9 video), a crease, higher price and more fragile · ★ large standard phone: cheaper, more durable, video fills the wide screen · for mostly watching shows, a large standard phone is usually the better value.

#### Q16 Ready *(applied typo fixes: "excessive", "preserving")*
> What's a good approach to lose exessive fat while presering the muscle mass I have when I can only exercise for 1 hour a day?

**Answer key (draft):** ★ a moderate calorie deficit (slow, steady loss; no crash diet) · ★ high protein intake · ★ prioritise resistance training in the hour (progressive overload), with some cardio or daily steps · sleep and recovery; see a doctor if there are health conditions.

### Writing: requested format, tone, length and content

#### Q17 Ready *(applied: "follow-up" → "follow-up email", so the format is stated)*
> Write a formal follow-up to my client for the business website proposal I sent him a week ago.

**Answer key (draft):** ★ formal, professional tone · ★ refers to the website proposal sent a week ago · ★ a clear, polite next step (questions, feedback, a call) · concise; placeholders for names.

#### Q18 Ready
> Write a five-bullet product description for an ergonomic office chair. Mention its durability and the lumbar support it offers.

**Why it's good:** the constraints are countable, so we see at once whether the F1 style breaks the format.
**Answer key (draft):** ★ exactly five bullets · ★ mentions durability · ★ mentions lumbar support · reads as a product description.

#### Q19 Ready
> Explain how end-to-end encryption works to a 10-year-old using fewer than 100 words.

**Why it's good:** the word limit and the style pull in opposite directions; code can count the words.
**Answer key (draft):** ★ fewer than 100 words · ★ correct idea: the message is locked on the sender's device and only the recipient's key opens it; the company in the middle can't read it · ★ language a 10-year-old understands.

### Technical: does the style damage code or technical accuracy?

#### Q20 Ready
> What is the difference between supervised and unsupervised machine learning? Give one example for each.

**Answer key (draft):** ★ supervised learns from labelled examples (inputs with correct answers) to predict labels · ★ unsupervised finds structure in unlabelled data · ★ one correct example each, e.g. spam filtering or house-price prediction; customer segmentation with k-means clustering.

#### Q21 Fixed
> Write a Python function that checks for specific keywords used in a pdf document.

**Issue:** inputs, outputs and library are open, so many different functions are "right" and a grader can't check them consistently.
**Final wording (approved):** *Write a Python function that takes a PDF file path and a list of keywords and returns the keywords that appear in the document, ignoring case. Use the pypdf library.*
**Answer key (draft):** ★ opens the file with `pypdf.PdfReader` and reads **every** page's text with `extract_text()` · ★ case-insensitive comparison · ★ returns the keywords found · handles pages with no extractable text; notes that scanned PDFs need OCR.

#### Q22 Fixed
> Write a SQL query that counts how many orders were placed last month and how many were completed. These are stored in a table containing order_id and client_id.

**Issue:** as written it can't be done. The table has no date column (so no "last month") and no status column (so no "completed").
**Final wording (approved):** *Write a PostgreSQL query that counts how many orders were placed last month and how many of those were completed. The table `orders` has the columns order_id, client_id, order_date and status.*
**Answer key (draft):** ★ filters to the previous calendar month (e.g. `order_date >= date_trunc('month', CURRENT_DATE) - INTERVAL '1 month' AND order_date < date_trunc('month', CURRENT_DATE)`) · ★ counts all orders and, conditionally, completed ones (`COUNT(*) FILTER (WHERE status = 'completed')` or `SUM(CASE …)`) · ★ one valid query · states the assumed status value.
*(Alternative: keep the original wording and move it to Uncertainty. A good answer would then point out the missing columns.)*

### Uncertainty: does the model invent, correct false premises, or flag the unknowable?

#### Q23 Ready
> What did Alan Turing do wrong that he was jailed?

**Why it's good:** a double false premise. He was never jailed, and he did nothing wrong by today's standards.
**Answer key (draft):** ★ corrects the premise: Turing was not imprisoned · ★ in 1952 he was convicted of "gross indecency" for a relationship with a man (homosexuality was then illegal in the UK) and accepted hormone treatment ("chemical castration") instead of prison · ★ the conviction is now seen as unjust: a government apology (2009) and a posthumous royal pardon (2013).

#### Q24 Ready
> Can you tell me the exact water temperature in Nice at 3 PM on August 7, 2027?

**Answer key (draft):** ★ says the exact future value cannot be known · may give the typical range for early August (the sea at Nice is usually mid-20s °C) · suggests checking a forecast or sea-temperature service nearer the date.

#### Q25 Ready
> What is the best AI model in the world?

**Note:** the judge is itself an AI model (Claude). The key rewards **how** the answer handles the question, not which model it names.
**Answer key (draft):** ★ says there is no single objective "best"; it depends on the task (coding, writing, cost, speed, open vs closed) · ★ notes that rankings change quickly and the model's own knowledge has a cutoff · may name leading models as examples without crowning one; suggests current leaderboards or testing on your own task.

---

## 4. What happened next

1. **18 Oct 2026:** I approved all the fixes above, including the typo fixes and moving Q8 to the uncertainty category.
2. The approved wording is in `data/authoring/my_questions.txt`: 25 questions, checked with the pipeline's own parser (procedure 3, quantitative 4, factual 2, explanation 3, advice 3, writing 3, technical 3, uncertainty 4; the most similar pair has 0.23 word overlap, well below the 0.5 duplicate threshold).
3. `scripts/eval_set/assemble_eval_set.py add-handwritten` drafted formal answer keys with Claude Opus 5.5. They went on the review sheet next to the drafts above, and I verified every one.
4. The evaluation set was then frozen with SHA-256 fingerprints. These 25 questions never change, and no training example may resemble them (checked automatically when the training data was built).
