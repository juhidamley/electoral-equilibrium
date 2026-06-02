# Survey Source Notes
*Generated 2026-06-01. Updated 2026-06-01 to add labeled readable files, fix pyreadstat status, correct NEP column schema, and flag open data-quality issues.*

> **Readable formats:** Each source now has a labeled Parquet (fast, typed) and CSV (inspectable) alongside the raw binary. Load with `pd.read_parquet(...)`. See the "Readable files" row in each source's table.

> **Source mapping note:** The sources named in the devplan map to actual downloaded files as follows:
> - "ARDA" → **ANES CDF** (American National Election Studies CDF, not the Association of Religion Data Archives)
> - "GSS" → **GSS** ✓
> - "Gallup" → **Democracy Fund VOTER Panel** (not Gallup)
> - "NEP" → **CNN/SSRS National Exit Polls** ✓
> - "Pew" → **NPORS 2024** (Pew Research Center) ✓
> - **CES** (Cooperative Election Study) — newly added, two files: cumulative 2006–2024 and 2024 single year

---

## 1. GSS — General Social Survey (NORC)

| | |
|---|---|
| **File** | `data/surveys/GSS_stata (1)/gss7224_r3.dta` |
| **Format** | Stata `.dta` (570 MB) |
| **Rows** | 75,699 respondents |
| **Variables** | 6,942 |
| **Years** | 1972–2024 (biennial; gaps in 1979, 1981, no 2019–2020) |
| **Codebook** | `GSS 2024 Codebook R3.pdf` |
| **Readable files** | `gss_labeled_subset.parquet` (1 MB) / `gss_labeled_subset.csv` (12 MB) — 34 columns, value labels applied, composite `weight` column |

### Year distribution (selected)
| Period | Annual N | Notes |
|---|---|---|
| 1972–1993 | ~1,400–1,600 | Annual except some off-years |
| 1994–2004 | ~2,800–3,000 | Two-ballot split samples begin |
| 2006 | 4,510 | Oversample year |
| 2008–2018 | ~2,000–2,900 | Biennial |
| 2021 | 4,032 | Post-COVID restart; no 2019 or 2020 surveys |
| 2022 | 3,544 | |
| 2024 | 3,309 | |

### Race / Ethnicity variables
| Variable | Label | Codes |
|---|---|---|
| `race` | Race (3-cat, all years) | 1=White, 2=Black, 3=Other |
| `racecen1/2/3` | Census-style multiselect (2000+) | 1=White, 2=Black/AA, 3=AIAN, 4–10=Asian subtypes, 14=Other PI, 15=Some other race, 16=Hispanic |
| `hispanic` | Hispanic identity | Numeric codes (country of origin) |
| `raceself` | Self-identified race (3-cat) | 1=White, 2=Black, 3=Other |

**Limitation:** `race` collapses all non-Black minorities into "other." For the five-stratum model, combine `racecen1 + hispanic` (available ~2000+).

### Religion variables
| Variable | Label | Codes |
|---|---|---|
| `relig` | Religious preference | 1=Protestant, 2=Catholic, 3=Jewish, 4=None, 5=Other, 6=Buddhism, 7=Hinduism, 8=Other Eastern, 9=Muslim/Islam, 10=Orthodox, 11=Christian, 13=Inter-nondenominational |
| `fund` | Protestant fundamentalism | 1=Fundamentalist, 2=Moderate, 3=Liberal |
| `attend` | Service attendance | 0=Never … 8=Several/wk |
| `reborn` | Born-again experience | Yes/No |
| `evangclx` | Born-again or evangelical (2018+) | — |
| `religid` | Denominational identity | 1=Fundamentalist, 2=Evangelical, 3=Mainline, 4=Liberal, 5=None, 6=Other |
| `pray` | Prayer frequency | — |
| `religimp` | Importance of religion | 1=Very … 4=Not at all |

### Gender variable
| Variable | Label |
|---|---|
| `sex` | 1=Male, 2=Female |
| `sexnow` / `sexnow1` | Current gender (2021+): Male, Female, Transgender, Other |

### Party / Political variables
| Variable | Label | Codes |
|---|---|---|
| `partyid` | Party ID (7-pt) | 0=Strong Dem, 1=Weak Dem, 2=Ind-Dem, 3=Independent, 4=Ind-Rep, 5=Weak Rep, 6=Strong Rep, 7=Other |
| `polviews` | Liberal–conservative (7-pt) | 1=Extremely liberal … 7=Extremely conservative |
| `pres68`–`pres20`, `whovote24` | Presidential vote recall | Per-year candidate codes |

### Validity assessment
| Dimension | Notes |
|---|---|
| **Methodology** | Area probability sample; in-person CAPI interviews. Highest-quality sampling design among these sources — addresses randomness and coverage rigorously. Since 2021, some questions moved to self-administered web supplement, which introduces a minor mode break. |
| **Response rate** | Declining: ~70–75% in 1970s, ~50–55% by 2018. Non-response bias increasingly a concern; lower-income, less-educated respondents harder to reach. NORC applies non-response weights to correct partially. |
| **Funder** | National Science Foundation (NSF). Federal grant funding — no private ideological sponsor. |
| **Ideological skew** | None documented. NORC (National Opinion Research Center, University of Chicago) is a nonpartisan survey research center with ~80 years of academic independence. Widely accepted as gold standard for social science time-series data. |
| **Known issues** | Two-year gap (2019–2020) due to COVID. Split-ballot design (50% of sample per form from 1994 onward) halves effective N for any given item. `race` variable (3-cat) insufficient for five-stratum model before 2000. UnicodeWarning in pandas: file uses Latin-1 encoding internally. |
| **Recommended use** | Best longitudinal source for religion × vote since 1972. Presidential vote recall available 1968–2024. Use `racecen1 + hispanic` for five-stratum race split (2000+). |

---

## 2. ANES CDF — American National Election Studies Cumulative Data File

> **Name note:** Called "ARDA" in the devplan. The actual data is from ANES (electionstudies.org), not the Association of Religion Data Archives.

| | |
|---|---|
| **File** | `data/surveys/anes_timeseries_cdf_csv_20260205/anes_timeseries_cdf_csv_20260205.csv` |
| **Format** | CSV (156 MB) |
| **Rows** | 73,745 respondents |
| **Variables** | 1,030 (VCF codes) |
| **Years** | 1948–2024 (biennial presidential-year focused) |
| **Readable files** | `anes_labeled_subset.parquet` (1 MB) / `anes_labeled_subset.csv` (14 MB) — 20 columns renamed to descriptive names, value labels applied |
| **Label dictionary** | `rawdata/anes_vcf_labels.json` — all 1,029 VCF codes extracted from codebook PDF; used to build the labeled subset. Auto-parsed; spot-check VCF0105a/VCF0128/VCF0704 against source PDF before production use. |

### Year distribution
| Period | N/year | Notes |
|---|---|---|
| 1948–1970 | 662–1,899 | Face-to-face, probability sample |
| 1972–2004 | 1,212–2,705 | Expanded race/ethnicity from 1972 |
| 2008 | 2,322 | Added internet component |
| 2012 | 5,914 | Dual-mode F2F + web |
| 2016 | 4,270 | |
| 2020 | 8,280 | Mostly web/telephone (COVID) |
| 2024 | 5,521 | |

### Key demographic variables
| Variable | Description | Values |
|---|---|---|
| `VCF0004` | Year | 1948–2024 |
| `VCF0104` | Gender | 1=Male, 2=Female |
| `VCF0105a` | Race (5-cat, 1966+) | 1=White non-Hisp, 2=Black non-Hisp, 3=Asian/PI, 4=Native Am, 5=Hispanic, 6=Hisp+Black, 7=Other |
| `VCF0106` | Race (4-cat, all years) | 1=White, 2=Black, 3=Other |
| `VCF0107` | Hispanic origin | 1=Mexican, 2=Puerto Rican, 3=Cuban, 4=Other Hisp; 7=Not Hisp |
| `VCF0128` | Religion (4-cat) | 0=No pref, 1=Protestant, 2=Catholic, 3=Jewish, 4=Other/none |
| `VCF0130` | Religious attendance | 0=Never … 5=More than weekly |
| `VCF0301` | Party ID (7-pt) | 1=Strong Dem … 7=Strong Rep |
| `VCF0803` | Ideology (7-pt) | 1=Extremely liberal … 7=Extremely conservative |
| `VCF0704` | Presidential vote | 0=Didn't vote, 1=Dem, 2=Rep, 3=Other |

### Validity assessment
| Dimension | Notes |
|---|---|
| **Methodology** | Face-to-face probability sample 1948–2012; dual-mode (F2F + web) 2016+; web-dominant 2020+. Multi-stage area probability design — strong methodological pedigree. Mode shift in 2020 introduces a structural break in time-series comparisons. |
| **Response rate** | High historically (70–80% pre-2000). Post-2012 internet component has lower response rates but broader demographic reach. |
| **Funder** | National Science Foundation (NSF) primarily, with some supplemental DARPA funding in early years. No private ideological funder. |
| **Ideological skew** | None documented. ANES is run jointly by University of Michigan and Stanford — longstanding academic partnership with no partisan bias record. The survey is the canonical academic source for vote behavior research. |
| **Known issues** | VCF codes are opaque — every variable requires codebook lookup. Religion only 4-cat (VCF0128); can't separate evangelical from mainline Protestant without `VCF0129` denomination codes (available only some years). No 2006, 2010, 2014, 2018 (CDF focuses on presidential years). Civic-engagement self-selection: the ANES is a multi-hour commitment, selecting for politically interested respondents. |
| **Recommended use** | Best longitudinal source (1948+) for race × party × vote. `VCF0105a` gives 5-cat race from 1966. Critical limitation: no evangelical/mainline split without supplemental denominational data. |

---

## 3. Democracy Fund VOTER Panel

> **Name note:** Called "Gallup" in the devplan. The actual data is from the Democracy Fund + UCLA VOTER Study Group (not Gallup).

| | |
|---|---|
| **File** | `data/surveys/VOTER Panel Data Files/voter_panel.csv` (40 MB) |
| **Format** | CSV (also `.dta` 91 MB and `.sav` 38 MB) |
| **Rows** | ~8,000 panel respondents |
| **Variables** | 1,797 columns |
| **Waves** | 2011, 2016, 2017, 2018, 2019Jan, 2019Nov, 2020Sep, 2020Nov |

### Variable structure
All variables: `{variable}_{wave}`. Demographic columns per wave: `race_{wave}`, `religion_{wave}`, `gender_{wave}`, `educ_{wave}`. Presidential vote: `presvote_{wave}` for 2012, 2016, 2019 (intent), 2020.

### Validity assessment
| Dimension | Notes |
|---|---|
| **Methodology** | Online opt-in panel recruited by YouGov using quota and stratified sampling matched to population benchmarks — NOT a probability sample. Respondents are re-surveyed across multiple waves (panel design). |
| **Response rate / attrition** | Panel attrition is the main concern. Respondents who stay through 8 waves (2011–2020Nov) are highly self-selected for political engagement. Missing data rates increase in early waves (2011 has fewer active respondents than 2020). |
| **Funder** | Democracy Fund (founded by Pierre Omidyar, eBay founder). Democracy Fund is explicitly nonpartisan but focuses on "strengthening democracy" — its funded projects include voter registration and election administration reform, which critics characterize as center-left priorities. UCLA provides academic oversight. |
| **Ideological skew** | Low concern for the demographic variables. The funder's democracy-strengthening mission could affect question framing on voting and civic engagement items, but race/religion/gender measures are standard. No documented ideological skew in the political variables. |
| **Known issues** | No visible respondent ID at the head of the CSV (first column is a weight). Missing values encoded as `__NA__` strings rather than NaN. Only covers 2011–2020; no 2022 or 2024 data. **Race and religion numeric codes are not decoded for most waves** — text labels (`race_t_{wave}`, `religion_t_{wave}`) exist only for 2016 and 2019Nov; all other waves require a per-wave codebook not yet in the repo. No labeled subset was built for this source. Panel conditioning: later-wave respondents have been surveyed many times and may exhibit fatigue or anchoring. |
| **Recommended use** | Primary source for computing individual-level Σ_Δ covariance from matched longitudinal records. The only source here that allows within-respondent change estimation. Do not use as a cross-sectional reference — attrition bias is severe. |

---

## 4. NPORS 2024 — Pew Research Center

| | |
|---|---|
| **File** | `data/surveys/NPORS-2024-Data-Release/NPORS_2024_for_public_release.sav` |
| **Format** | SPSS `.sav` (658 KB) |
| **N** | 5,626 respondents |
| **Field dates** | Feb 1 – Jun 10, 2024 |
| **Sample design** | National address-based sample (ABS) from USPS Delivery Sequence File |
| **Modes** | Online, Paper, Telephone |
| **Languages** | English and Spanish |
| **Readable files** | `npors_2024_labeled.parquet` (240 KB) / `npors_2024_labeled.csv` (7 MB) — all 79 columns, value labels applied via pyreadstat |

### Key variables (from SPSS binary header)
`RESPID`, `MODE`, `LANGUAGE`, `RELIG`, `BORN`, `RELIMP`, `PRAY`, `PARTY`, `PARTYLN`, `HISP`, `RACECMB`, `RACETHN`, `AGE`, `AGECAT`, `GENDER`, `MARITAL`, `EDUCCAT`, `CREGION`, `METRO`, `BASEWT`, `WEIGHT`

### Validity assessment
| Dimension | Notes |
|---|---|
| **Methodology** | Address-based sample (ABS) — considered the gold standard for coverage. Physical addresses from the USPS file cover >97% of US households, including non-internet households. Multi-mode design (mail, web, phone) reduces mode-related coverage bias. This is Pew's own "reference" survey, designed specifically to serve as a calibration benchmark. |
| **Response rate** | Address-based mail surveys typically yield 10–20% response rates, but Pew uses extensive follow-up and incentives. The long field period (5 months) is by design to reach hard-to-contact populations. |
| **Funder** | Pew Charitable Trusts (private foundation established by heirs of Sun Oil Company founder J.N. Pew). Pew Research Center spun off as an independent subsidiary in 2004 with an explicit nonpartisan mission. Annual budget ~$100M. |
| **Ideological skew** | Generally considered nonpartisan by academics and both major parties. Some conservative religious organizations have criticized Pew's religion surveys for question framing they view as secular-leaning. However, the NPORS is methodologically designed to be the most unbiased possible reference point for religion × demographics. No documented systematic partisan bias in published accuracy evaluations. |
| **Known issues** | Single cross-section (2024 only). Long field period (Feb–Jun 2024) means political opinions were measured across a 5-month window that included the Biden-Trump debate and Pres. Biden's announcement to withdraw. **`vote_2024_pres` (VOTEGEN_POST) shows Biden as the Democratic candidate** — field ended June 10, 2024, seven weeks before Biden's July 21 withdrawal. Do not use as ground truth for the 2024 election; use as a 2020 recall proxy or pre-withdrawal 2024 sentiment only. SPSS format now readable via pyreadstat==1.2.0 (pinned — do not upgrade, 1.3.x breaks Python 3.9). |
| **Recommended use** | Best single-year calibration benchmark for Pew-style religion × race × party marginals. The `RELIG + BORN + RELIMP + PRAY` combination covers all 7 religion strata. Use as external validation target for GSS and CES estimates. |

---

## 5. CES — Cooperative Election Study (Harvard Dataverse)

Two files have been added, covering different scopes:

### 5a. CES Cumulative 2006–2024

| | |
|---|---|
| **File** | `data/surveys/CES_2006_2024/cumulative_2006-2024.feather` (134 MB, fastest) |
| **Also available** | `cumulative_2006-2024.dta` (675 MB Stata), `cumulative_2006-2024.rds` (38 MB R) |
| **Guide** | `guide_cumulative_2006-2024.pdf` |
| **Readable files** | `ces_cumulative_labeled.parquet` (49 MB) / `ces_cumulative_labeled.csv` (599 MB) — all 109 columns, numeric value codes mapped to strings. Use Parquet; the CSV is provided for inspection only. |
| **Citation** | Kuriwaki, Shiro (2025). Cumulative CES Common Content. doi:10.7910/DVN/II2DB6, Harvard Dataverse V11. |
| **Rows** | 701,955 respondents |
| **Variables** | 109 (standardized, harmonized across years) |
| **Years** | 2006–2024 (annual; ~60,000/year in even years, ~15,000–26,000 in odd years) |
| **Survey vendor** | YouGov |

#### Year distribution
| Year | N | Year | N |
|---|---|---|---|
| 2006 | 36,421 | 2016 | 64,600 |
| 2007 | 9,999 | 2017 | 18,200 |
| 2008 | 32,800 | 2018 | 60,000 |
| 2009 | 13,800 | 2019 | 18,000 |
| 2010 | 55,400 | 2020 | 61,000 |
| 2011 | 20,150 | 2021 | 25,700 |
| 2012 | 54,535 | 2022 | 60,000 |
| 2013 | 16,400 | 2023 | 24,500 |
| 2014 | 56,200 | 2024 | 60,000 |
| 2015 | 14,250 | **Total** | **701,955** |

#### Race / ethnicity variables
| Variable | Description | Codes / counts |
|---|---|---|
| `race` | Self-identified race (all years) | 1=White (510K), 2=Black (80K), 3=Hispanic (60K), 4=Asian (17K), 5=Native Am (5.8K), 6=Mixed (15K), 7=Other (11K), 8=Middle Eastern (1.2K) |
| `race_h` | Any-part Hispanic (preferred) | Reclassifies White/Black Hispanics → Hispanic; 1=White (499K), 2=Black (79K), 3=Hispanic (80K), 4=Asian (17K), 5=Native Am (5.2K) |
| `hispanic` | Hispanic identity follow-up | Yes (30K), No (538K); asked of non-Hispanic race respondents |
| `hisp_origin` | Hispanic country/region of origin | Mexico, US, Spain, Puerto Rico, South America, Cuba, etc. (2015+) |

**Note:** Use `race_h` for the five-stratum model — it correctly classifies any-part Hispanics regardless of how they answered the race question.

#### Religion variables
| Variable | Description | Values |
|---|---|---|
| `religion` | Religious affiliation | 1=Protestant (241K), 2=Roman Catholic (134K), 3=Mormon (9.3K), 4=Orthodox (3.5K), 5=Jewish (17K), 6=Muslim (4K), 7=Buddhist (5.7K), 8=Hindu (2K), 9=Atheist (37K), 10=Agnostic (40K), 11=Nothing in Particular (125K), 12=Something Else (45K) |
| `relig_bornagain` | Born-again or evangelical Christian | Yes (200K), No (482K); available all years except 2007 |
| `relig_imp` | Importance of religion | Very Important (249K), Somewhat (167K), Not Too (97K), Not at All (136K) |
| `relig_church` | Church attendance | More than once/week (54K), weekly (115K), 1–2×/month (50K), few/year (90K), seldom (150K), never (185K) |
| `relig_protestant` | Protestant denomination | Baptist (76K), Nondenominational (54K), Methodist (34K), Lutheran (26K), Presbyterian (16K), Pentecostal (17K), Episcopalian (11K), etc. |

**For our evangelical stratum:** `relig_bornagain == "Yes"` is the cleanest identifier. Available 2006+ (except 2007).

#### Gender variable
| Variable | Years | Codes |
|---|---|---|
| `gender` | All years (standardized binary) | 1=Male, 2=Female |
| `sex` | 2006–2020 | Male, Female |
| `gender4` | 2021+ | Man (77K), Woman (92K), Non-Binary (1.3K), Other (328) |

#### Party / political variables
| Variable | Description | Values |
|---|---|---|
| `pid3` | Party ID (3-pt) | 1=Democrat (258K), 2=Republican (184K), 3=Independent (197K), 4=Other (28K), 5=Not Sure (34K) |
| `pid7` | Party ID (7-pt) | 1=Strong Dem (174K) … 7=Strong Rep (119K); 4=Pure Independent (98K) |
| `pid3_leaner` | Party ID including leaners | Dem incl. leaners (328K), Rep incl. leaners (253K), Pure Ind. (98K) |
| `ideo5` | Ideology (5-pt) | Very Liberal (73K), Liberal (125K), Moderate (221K), Conservative (151K), Very Conservative (80K) |

#### Vote variables
| Variable | Description |
|---|---|
| `intent_pres_08/12/16/20/24` | Pre-election presidential preference (candidate-level) |
| `voted_pres_08/12/16/20/24` | Post-election presidential vote choice (candidate-level) |
| `voted_pres_party` / `intent_pres_party` | Rolled-up party variable |
| `vv_turnout_gvm` | **Validated turnout** matched to voter files (Voted: 312K; No record: 227K) |
| `vv_regstatus` | Validated registration status |
| `vv_party_gen` | Validated registered party (where available by state) |

#### Weights
| Weight | Description |
|---|---|
| `weight` | Year-specific pre-election weight (representative of national adults) |
| `weight_cumulative` | `weight` divided by year-specific factor — makes years comparable for pooled analysis |
| `weight_post` | Post-election wave weight (2012, 2016, 2018, 2020, 2022 only) |
| `vvweight` | Weight to validated registered voter population (2018–2024) |

---

### 5b. CES 2024 Single-Year

| | |
|---|---|
| **File** | `data/surveys/CES_2024/CCES24_Common_OUTPUT_vv_topost_final.csv` (175 MB) |
| **Also available** | `.dta` (947 MB Stata) |
| **Questionnaires** | `CCES24_Common_pre.docx`, `CCES24_Common_post.docx` |
| **Guide** | `CES_2024_GUIDE_vv.pdf` |
| **Variables** | 694 columns |
| **Key vars not in cumulative** | `CC24_*` policy questions, detailed denomination breakdowns (`religpew_baptist`, `religpew_pentecost`, etc.), `pew_bornagain`, `pew_churatd`, `pew_prayer`, `mena` (MENA identity), `CC24_hisp_*` / `CC24_asian_*` / `CC24_native` (detailed ethnic origin), validated 2024 vote: `TS_g2024`, `TS_voterstatus` |
| **Readable files** | `ces_2024_labeled.parquet` (1.4 MB) / `ces_2024_labeled.csv` (9 MB) — 21 key columns, value labels applied, descriptive names |
| **⚠ Open issue** | `pres_vote_2024` (from `CC24_410`) identified as presidential vote by question-numbering convention. Not yet verified against `CES_2024_GUIDE_vv.pdf`. Confirm before using in production. |

---

### CES validity assessment (applies to both 5a and 5b)
| Dimension | Notes |
|---|---|
| **Methodology** | **Online matched sample** — NOT a traditional probability sample. YouGov maintains a large opt-in internet panel (~2M+ US members). For each year's CES, they draw a stratified random subsample from the panel matched to target population benchmarks (Census race/ethnicity, sex, age, education, state) using propensity score matching. This "sample matching" is methodologically controversial. |
| **Response rate** | Conceptually not applicable in the traditional sense (YouGov panel members are recruited for future surveys; retention rates are undisclosed). The matched-sample approach sidesteps response rate as a validity metric, substituting covariate balance instead. |
| **Validated turnout** | Major methodological strength: YouGov matches respondents' PII to commercial voter files (Catalist 2006–2020; TargetSmart 2022+) to validate whether they actually voted. This eliminates social-desirability inflation in self-reported turnout. Available 2006+ for general election. Voter file matching has limitations: unmatched respondents are conservatively coded as "no record of voting," which slightly understates turnout. |
| **Funder** | Consortium of dozens of academic institutions, each paying for "team content" modules on top of shared common content. No single private sponsor. YouGov provides infrastructure. Principal investigators: Brian Schaffner (Tufts), Jeremy Pope (BYU), Marissa Shih (YouGov). |
| **Ideological skew** | No single funder bias. The university consortium spans ideological spectrum (e.g., BYU to Berkeley). YouGov as vendor has no documented ideological agenda. However, the opt-in panel and online-only delivery over-represents politically engaged, higher-educated respondents — residual after weighting but attenuated. Multiple published validation studies show CES estimates correlate well with official election results at the state level. |
| **Known biases** | (1) **Online-only coverage gap**: ~10–15% of US adults lack broadband internet; this group skews older, lower-income, rural, and less educated — systematically excluded from the panel. (2) **Panel self-selection**: Politically disengaged respondents are harder to recruit and retain into YouGov's panel; estimates may overstate civic knowledge and political engagement. (3) **Democratic overestimate in some elections**: Published comparisons show CES vote shares occasionally overestimate Democratic performance (e.g., 2020 Wisconsin presidential margin), attributed to panel composition and weighting limitations. (4) **Hispanic undercounting**: The guide explicitly notes that the `race` (not `race_h`) variable undercounts Hispanics who answered Hispanic via the Hispanic follow-up question rather than the main race question — use `race_h`. (5) **Weight cumulative caveat**: `weight_cumulative` rescales each year to equal sample size; if pooling odd + even years, every even-year observation is down-weighted by ~2× automatically. |
| **Academic standing** | The CES is the most widely used dataset in US electoral research by volume of citations. The large annual N (~60K) makes it exceptional for subgroup analysis — the only source here that can reliably estimate vote share for small groups (e.g., Black evangelical women, Latino Catholics) in a single year. Validated turnout from voter files is unique among these sources. |
| **Recommended use** | Primary source for constructing demographic × vote crosstabs for the five-stratum model. Use `race_h` + `relig_bornagain` + `gender`. Annual coverage 2006–2024 provides sufficient history for the Mistral fine-tuning dataset. For covariance estimation, combine with VOTER Panel (individual-level change). |

---

## 6. NEP — National Election Pool Exit Polls (CNN/SSRS)

| | |
|---|---|
| **Files** | `data/cnn_ssrs_polls/nep_{year}_exit_poll.{csv,json}` |
| **Raw PDFs** | `data/surveys/nep_{2004,2016,2020,2024}.pdf` |

### Available years and sample sizes
| Year | N | Rows | Candidate matchup |
|---|---|---|---|
| 2004 | 13,660 | 229 | Bush vs. Kerry (vs. Nader) |
| 2016 | 24,558 | 131 | Clinton vs. Trump (vs. Other) |
| 2020 | 15,590 | 251 | Biden vs. Trump |
| 2024 | 22,966 | 225 | Harris vs. Trump |

### CSV column schema
`category`, `n_total`, `sub_category`, `sub_pct`, `dem_candidate`, `rep_candidate`, `dem_pct`, `rep_pct` (+ `other_pct` for 2016 only)

Note: schema was updated from the original `candidate`/`candidate_pct`/`trump_pct` to symmetric party-based names. `trump_pct` was incorrect for 2004 (Bush). The Republican candidate is now auto-detected from the PDF and stored in `rep_candidate`.

**⚠ Open issue:** `category` and `sub_category` are free-text labels that vary across years ("Vote by race" vs "Race" vs "VOTE BY RACE"). A lookup table mapping these to canonical `stratum` and `bloc` IDs (`african_american`, `evangelical`, etc.) is documented in `rawdata/column_maps.json` but not yet built. Nothing downstream can consume these files until that mapping exists.

### Demographic categories available (varies by year)
Race, Religion, Age, Gender, Party ID, Ideology, Education, Income, Region, Marital status, Union household, First-time voter, abortion attitude, candidate quality questions.

### Validity assessment
| Dimension | Notes |
|---|---|
| **Methodology** | **Exit polls**: in-person surveys of actual voters leaving polling stations, plus telephone surveys of early/absentee voters. The sample is inherently a convenience/intercept sample within precincts, not a random probability sample of the electorate. Precincts are selected with probability proportional to size. Within precincts, interviewers approach every nth voter (systematic selection). |
| **Response rate** | Historically 40–60% of approached voters agree to complete the questionnaire. Non-response is the primary validity concern: voters who decline may differ systematically from those who participate. After 2016, CNN/SSRS shifted toward a "telephone-first" design for early voters, creating a dual-mode structure with potential mode effects. |
| **Funder** | National Election Pool consortium: ABC News, CBS News, CNN, Fox News, NBC News. SSRS (Social Science Research Solutions) is the contracted field organization. Commercial media organizations have financial incentives for accurate race calls on election night, which aligns their interests with methodological rigor. |
| **Ideological skew** | CNN is a member of the pool — some characterize CNN as having a center-left editorial bias — but the exit poll methodology and field organization (SSRS) are separate from editorial decisions. The pool shares identical questionnaires across all five networks; no single network controls the methodology. SSRS is a nonpartisan research firm. |
| **Known historical failures** | The 2004 exit polls famously showed Kerry winning (by a significant margin in afternoon wave) before results diverged — attributed to differential response rates (Republican-leaning voters refused at higher rates). This led to major methodology reforms, including later release times and the addition of absentee/early voter phone surveys. |
| **Known biases** | (1) **Differential partisan refusal**: Republican-leaning voters have historically been less willing to complete exit poll questionnaires, potentially understating Republican performance in pre-result estimates (though weighting attempts to correct). (2) **Mode effects**: Phone surveys of early voters may differ systematically from in-person exit polls of Election Day voters. (3) **Sample-voters only**: NEP samples actual voters, not registered or likely voters — this is a feature for vote-share estimation but means the dataset cannot speak to non-voters or turnout rates. (4) **No cross-tabulation**: Each row shows a single demographic group's vote share; cannot compute joint distributions (e.g., Black evangelical women) from exit poll data alone. |
| **Academic standing** | Exit polls are widely used as the closest approximation to actual vote behavior broken down by demographics. They are the primary source for understanding "who voted for whom" in any given election cycle. The large N and validated-voter population (actual voters) make them more reliable than pre-election surveys for this specific task. However, academic researchers typically use them for historical description, not model estimation, because of selection and weighting concerns. |
| **Recommended use** | Use as the target output variable (μ_race, μ_religion, μ_gender vote shares) for model calibration. Also use as a cross-year benchmark to validate GSS and CES estimates. Do not use for Σ_Δ estimation (no individual-level change data). Note: 2008 and 2012 exit poll data not yet acquired. |

---

## Cross-source comparison

| Dimension | GSS | ANES CDF | VOTER Panel | NPORS/Pew | CES Cumulative | NEP |
|---|---|---|---|---|---|---|
| **Years** | 1972–2024 | 1948–2024 | 2011–2020 | 2024 only | 2006–2024 | 2004/16/20/24 |
| **N per election year** | ~2,000–3,000 | ~2,000–8,300 | ~8,000 (panel) | 5,626 | ~60,000 | 13,000–25,000 |
| **Population** | US adults | US citizens | US voters (panel) | US adults (ABS) | US adults (online panel) | Actual voters |
| **Sampling method** | Probability (area) | Probability (area) | Matched online panel | Address-based (probability) | Matched online panel | Precinct intercept |
| **Race 5-stratum** | 2000+ only | 1966+ (VCF0105a) | All waves (codes TBD) | RACECMB+HISP | `race_h` all years | Yes (direct) |
| **Religion 7-stratum** | Strong (relig+fund+attend+evangclx) | Weak (4-cat, VCF0128) | Wave-specific | Strong (RELIG+BORN+RELIMP) | Strong (religion+relig_bornagain) | Yes (direct) |
| **Gender 3-stratum** | `sex`; `sexnow` 2021+ | VCF0104 (2-cat only) | `gender_{wave}` | GENDER | `gender` / `gender4` | Yes (direct) |
| **Presidential vote** | pres68–pres20 + 2024 | VCF0704 | `presvote_{wave}` | No | `voted_pres_08–24` | Direct (primary output) |
| **Validated turnout** | No | No | No | No | Yes (voter files) | Yes (actual voters) |
| **Funder** | NSF (federal) | NSF (federal) | Democracy Fund (private) | Pew Trusts (private) | Academic consortium | Media consortium |
| **Sampling quality** | ★★★★★ | ★★★★★ | ★★★ | ★★★★★ | ★★★ | ★★★★ |

---

## Usage recommendations for this project

1. **Primary μ estimation (vote shares by stratum):** CES cumulative (`race_h`, `relig_bornagain`, `religion`, `gender`, `voted_pres_party`) is the best single source — large N allows reliable subgroup estimates for all three strata in every year 2006–2024. Cross-validate against NEP.

2. **Pre-2006 historical data:** GSS (`relig + fund + attend`, `racecen1 + hispanic`, `sex`, `partyid`, `presXX`) back to 1972; ANES CDF (`VCF0105a`, `VCF0128`, `VCF0104`, `VCF0301`, `VCF0704`) back to 1966/1948.

3. **Religion detail (evangelical):** CES `relig_bornagain` (cleanest, 2006+); GSS `evangclx` (2018+) or `fund=fundamentalist + relig=Protestant`; NPORS `RELIG+BORN` (2024 benchmark).

4. **Latino/Asian isolation:** CES `race_h` (2006+, recommended); ANES `VCF0105a` (1966+); GSS `racecen1 + hispanic` (2000+).

5. **Cross-sectional 2024 benchmark:** NPORS is the most methodologically rigorous 2024 source (address-based, multi-mode). CES 2024 (60K N) provides the most granular subgroup estimates.

6. **Covariance Σ_Δ estimation:** VOTER Panel for individual-level 2011–2020 shifts (matched respondents across waves). CES year-over-year changes as a cross-sectional proxy.

7. **Model output validation:** NEP exit polls — the ground truth for "who actually voted for whom" in 2004, 2016, 2020, 2024. Target for calibrating μ_eff estimates.

8. **Do not use for Σ_Δ:** GSS, ANES CDF, NPORS, NEP are all cross-sectional; no within-respondent change data.

---

## Data kernel run — gap analysis (2026-06-01)

Kernel: `electoral/kernels/data.py::build_voter_panel()`  
Config: `configs/base.json` · Run at: 2026-06-01

### Final resolved panel summary

| Metric | Value |
|---|---|
| Rows before conflict resolution | 305 |
| Rows after conflict resolution | 220 |
| Conflicts resolved (inverse-SE weighted) | 85 (cycle, bloc) pairs |
| Election cycles covered | 20 (1948–2024) |
| Canonical blocs with any coverage | 14 of 15 |
| Bloc with zero coverage | `evangelical` |

### Counts per source per cycle (raw, before resolution)

| Source | 2000 | 2004 | 2008 | 2012 | 2016 | 2020 | 2024 |
|---|---|---|---|---|---|---|---|
| ANES | 11 | 11 | 11 | 11 | 12 | 11 | 0 |
| CES | 0 | 0 | 13 | 13 | 13 | 13 | 13 |
| GSS | 0 | 0 | 0 | 0 | 13 | 13 | 0 |
| NEP | 0 | 0 | 0 | 0 | 0 | 6 | 9 |

Notes: ANES 2024 not in labeled subset. CES starts 2006. GSS only processes
pres16/pres20 retrospective columns. NEP 2004/2016 produce 0 rows due to
stratum name mismatch (see GAP-4, GAP-5).

### Coverage matrix — 2000–2024 presidential cycles

Sources present per (cycle, bloc): **A**=ANES · **C**=CES · **G**=GSS · **N**=NEP · `----`=no data

| Bloc | 2000 | 2004 | 2008 | 2012 | 2016 | 2020 | 2024 |
|---|---|---|---|---|---|---|---|
| african_american | A | A | AC | AC | ACG | ACGN | CN |
| asian | A | A | AC | AC | ACG | ACGN | CN |
| latino | A | A | AC | AC | ACG | ACGN | CN |
| white | A | A | AC | AC | ACG | ACGN | CN |
| other_race | A | A | AC | AC | ACG | ACG | CN |
| **evangelical** | **----** | **----** | **----** | **----** | **----** | **----** | **----** |
| catholic | A | A | AC | AC | ACG | ACG | CN |
| protestant | A | A | AC | AC | ACG | ACG | C |
| secular | ---- | ---- | C | C | CG | CG | C |
| jewish | A | A | AC | AC | ACG | ACG | CN |
| muslim | ---- | ---- | C | C | CG | CG | C |
| other_rel | A | A | AC | AC | ACG | ACG | C |
| women | A | A | AC | AC | ACG | ACGN | CN |
| men | A | A | AC | AC | ACG | ACGN | CN |
| **other_gender** | **----** | **----** | **----** | **----** | **----** | **----** | **----** |

### Gap catalog

#### GAP-1 · `evangelical` — ALL CYCLES · Severity: HIGH

No source currently provides `evangelical` as a standalone stratum.

| Source | Situation |
|---|---|
| ANES | 4-category religion (Protestant/Catholic/Jewish/Other). No evangelical sub-split in labeled subset. Raw CDF has `VCF0128`/`VCF0129` denomination + `VCF0130` attendance — not extracted. |
| GSS | `relig_bornagain` (`reborn`) and `fund` columns available in labeled subset but kernel `_from_gss` does not use them. `evangclx` available 2018+. |
| CES | `relig_bornagain` flag present for all 2006–2024 years (~200K "Yes" respondents). Not used in current `_from_ces` aggregation. Kernel `_CES_RELIGION` remap has "Evangelical Protestant" entry but CES `bloc__religion` values are standard denominational strings, not evangelical flags. |
| NEP | "white evangelical/born-again?" sub-stratum present in 2004 and 2020 PDFs but conditional on race (White only) — excluded to avoid double-counting. |

**Fix (CES, priority 1):** In `_from_ces`, add a second aggregation pass that filters
`relig_bornagain == "Yes"` (or the numeric equivalent), groups by (cycle), computes
weighted mean of `vote_indicator`, assigns `bloc = "evangelical"`. This gives
evangelical coverage for 2008–2024 from CES alone.

**Fix (GSS, priority 2):** In `_from_gss`, add evangelical extraction using
`reborn == "Yes"` (available 2000+) or `evangclx == "Yes"` (2018+).

#### GAP-2 · `other_gender` — ALL CYCLES · Severity: LOW (~1% of electorate)

All current sources use binary Male/Female coding. ANES 2016 has 11 "Other"
respondents (n=11, 100% Democrat, SE=0) — too sparse for production use.

CES `gender4` (2021+) includes "Non-Binary" (~1,300) and "Other" (~328) categories
but the kernel's `_CES_GENDER` remap currently only maps Male→men and Female→women.

**Fix:** Inspect CES `gender4` value labels for Non-Binary/Other. Add to `_CES_GENDER`
remap. Aggregation will produce `other_gender` bloc rows for 2024 from CES.

#### GAP-3 · `secular` and `muslim` missing 2000–2004 · Severity: MEDIUM

ANES covers 2000/2004 but maps "Other and none (also includes DK preference)"
→ `other_rel` (conservative assignment). This collapses secular and other-faith
respondents. ANES has no Muslim sub-category in the labeled subset.

CES starts 2006; GSS only covers 2016/2020.

**Fix (secular):** Reclassify ANES "Other and none" → `secular`. This gives secular
coverage back to 1948 (all ANES cycles). Caveat: some "Other" faiths (Hindu, Buddhist)
are also in this bucket; the misclassification is small (secular dominates the "none"
category in post-1990 data).

**Fix (muslim):** Add ANES denomination sub-code for Muslim to the `anes_labeled_subset`
extraction. Alternatively accept the gap for 2000/2004 given small Muslim sample in ANES.

#### GAP-4 · NEP 2004 — 0 rows extracted · Severity: MEDIUM

The 2004 CNN/SSRS exit-poll CSV uses stratum names "vote by race", "vote by religion",
"vote by gender" — not the "Race"/"Religion"/"Gender" strings the kernel expects.

**Recoverable data in the 2004 file:**
- Race ("vote by race"): White 41%, African-American 88%, Latino 53%, Asian 56%, Other 54%
- Religion ("vote by religion"): Protestant 40%, Catholic 47%, Jewish 74%, Other 74%, None 67%
- Gender ("vote by gender"): Male 44%, Female 51%
- Note: "white evangelical/born-again?" sub-stratum present (Yes=21%, No=56%)

**Fix:** In `_from_nep`, extend `is_race`/`is_religion`/`is_gender` filters to
case-insensitively match "vote by race", "vote by religion", "vote by gender" patterns.

#### GAP-5 · NEP 2016 — 0 rows extracted · Severity: MEDIUM

The 2016 exit-poll CSV uses lowercase stratum names "race" and "religion". Gender data
is embedded in the "national president" stratum (first two rows: male/female).

**Recoverable data:**
- Race ("race" stratum): white, black, non-white (limited — no Latino/Asian/Other breakdown)
- Religion ("religion" stratum): protestant, catholic, mormon — partial; Jewish/secular/muslim absent
- Gender ("national president" rows 0–1): male 42%, female 54%

**Fix:** Case-insensitive stratum matching. Race and gender data are recoverable.
Religion is partial; some rows appear OCR-corrupted in the PDF.

#### GAP-6 · ANES 2024 — 0 rows · Severity: LOW

`anes_labeled_subset.parquet` contains cycles [1948–2020]. ANES 2024 data was
available at the time of extraction but may not have been included in the labeled
subset generation. CES and NEP already cover 2024.

**Fix:** Check electionstudies.org for the 2024 ANES labeled release; re-run
labeled-subset extraction if available.

#### GAP-7 · GSS 2024 — 0 rows · Severity: LOW

`_GSS_ELECTIONS` only maps pres16/pres20. The GSS `whovote24` column gives 2024
retrospective presidential vote but is not in the current mapping.

**Fix:** Add `2024: ("whovote24", {"harris": 1.0, "trump": 0.0})` to `_GSS_ELECTIONS`
in `electoral/kernels/data.py`. Verify candidate-name strings in the labeled subset.

### Remediation priority

| Priority | Gap | Blocs affected | Cycles fixed | Effort |
|---|---|---|---|---|
| 1 | GAP-1 via CES `relig_bornagain` | `evangelical` | 2008–2024 (7 cycles) | ~30 min |
| 2 | GAP-4 NEP 2004 stratum names | race (5), religion (6), gender (2) | +2004 | ~1 hr |
| 3 | GAP-5 NEP 2016 case-insensitive | race (partial), gender (2) | +2016 | ~30 min |
| 4 | GAP-3 ANES secular reclassification | `secular` | +2000/2004 | ~15 min |
| 5 | GAP-7 GSS 2024 `whovote24` | all 13 GSS blocs | +2024 | ~15 min |
| 6 | GAP-2 CES `gender4` | `other_gender` | 2024 | ~30 min |
| 7 | GAP-1 via GSS `reborn`/`evangclx` | `evangelical` | +2016/2020 validation | ~30 min |
| 8 | GAP-6 ANES 2024 availability check | all 14 blocs | +2024 | ~15 min |
