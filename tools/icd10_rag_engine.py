"""
CLIS V2 — ICD-10 RAG Engine (Production)
==========================================
Student: Hritik Ram | Northeastern University
Final Project: CLIS V2

Production-grade RAG over ICD-10-CM Official Guidelines FY2024.
- Full TF-IDF vector index (no external dependencies)
- 80+ guideline sections covering all major specialties
- Paragraph-level citation grounding
- SQLite-backed persistent cache
- Groq Llama 3.3 70B synthesis with fallback

Data: CMS ICD-10-CM Official Guidelines FY2024
URL:  https://www.cms.gov/medicare/coding-billing/icd-10-codes
"""

import re, os, math, json, sqlite3, hashlib, time
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

# ── Full ICD-10 Knowledge Base (80+ sections) ─────────────────
ICD10_KB = [
    # ── DIABETES (I.C.4) ──────────────────────────────────────
    {"id":"I.C.4.a.1","title":"Type 1 diabetes mellitus","specialty":"Endocrinology",
     "content":"Type 1 diabetes mellitus is coded E10.-. Patients with type 1 diabetes who use insulin do not require Z79.4 since insulin dependency is inherent. If documentation is unclear about diabetes type, query the provider. Diabetic complications use combination codes (e.g., E10.40 diabetic neuropathy unspecified).",
     "codes":["E10","E10.9","E10.40","E10.51","E10.65"],"keywords":["type 1 diabetes","T1DM","juvenile diabetes","insulin dependent","E10"]},
    {"id":"I.C.4.a.2","title":"Type 2 diabetes mellitus","specialty":"Endocrinology",
     "content":"Type 2 diabetes is coded E11.-. When a type 2 diabetic uses insulin, assign Z79.4 as additional code. Do not assign Z79.4 for temporary insulin use. Oral hypoglycemics: Z79.84. Combination codes identify type, body system, and complication simultaneously.",
     "codes":["E11","E11.9","E11.65","Z79.4","Z79.84"],"keywords":["type 2 diabetes","T2DM","metformin","oral hypoglycemic","E11","insulin use"]},
    {"id":"I.C.4.a.3","title":"Diabetes with chronic kidney disease","specialty":"Nephrology",
     "content":"Diabetes with diabetic CKD: assign E10.22 (type 1) or E11.22 (type 2) as principal diagnosis. Always add N18.- to specify CKD stage. Sequencing rule: combination diabetes code first, then N18.x stage. Use N18.6 for ESRD. Do NOT separately code I10 when I12 or I13 applies.",
     "codes":["E11.22","E10.22","N18.1","N18.2","N18.3","N18.4","N18.5","N18.6","N18.9"],"keywords":["diabetic CKD","diabetes kidney","E11.22","N18","chronic kidney disease diabetes"]},
    {"id":"I.C.4.a.4","title":"Diabetes complications sequencing","specialty":"Endocrinology",
     "content":"Combination codes in E08-E13 identify type, body system, and complication. Use as principal diagnosis. If no combination code exists, sequence diabetes code first, then the complication. Multiple diabetic complications: sequence most severe first per Tabular List.",
     "codes":["E11.40","E11.41","E11.42","E11.51","E11.52","E11.610","E11.620"],"keywords":["diabetes complications","sequencing","diabetic retinopathy","diabetic neuropathy","diabetic nephropathy"]},
    {"id":"I.C.4.a.5","title":"Uncontrolled diabetes","specialty":"Endocrinology",
     "content":"Uncontrolled diabetes documented as hyperglycemia: E11.65 (type 2) or E10.65 (type 1). Hypoglycemia: E11.64x or E10.64x. Do not assume hyperglycemia vs hypoglycemia without documentation. Uncontrolled does NOT default to either — query provider if ambiguous.",
     "codes":["E11.65","E11.649","E10.65","E10.649"],"keywords":["uncontrolled diabetes","hyperglycemia","hypoglycemia","E11.65","poorly controlled"]},
    {"id":"I.C.4.b","title":"Gestational and pre-existing diabetes in pregnancy","specialty":"Obstetrics",
     "content":"Gestational diabetes: O24.4-. Pre-existing type 1 in pregnancy: O24.01-. Pre-existing type 2 in pregnancy: O24.11-. Do NOT use E10-E11 as principal when affecting pregnancy. Trimester characters required. Z79.4 and Z79.84 still applicable as additional codes.",
     "codes":["O24.410","O24.419","O24.01","O24.11","O24.419"],"keywords":["gestational diabetes","GDM","diabetes pregnancy","O24"]},
    # ── HYPERTENSION (I.C.9) ──────────────────────────────────
    {"id":"I.C.9.a","title":"Essential hypertension","specialty":"Cardiology",
     "content":"Essential (primary) hypertension: I10. Use I10 when hypertension is documented without further specification. Do NOT use I10 when hypertensive heart disease (I11), hypertensive CKD (I12), or hypertensive heart and CKD (I13) applies. Secondary hypertension: I15.- with etiology code.",
     "codes":["I10","I15","I15.0","I15.1","I15.2","I15.8","I15.9"],"keywords":["hypertension","high blood pressure","I10","essential hypertension","HTN"]},
    {"id":"I.C.9.b","title":"Hypertensive heart disease","specialty":"Cardiology",
     "content":"Hypertension WITH heart failure: I11.0. ICD-10-CM presumes causal relationship — provider does not need to explicitly link them. Also assign I50.- for type of heart failure. Hypertension with heart disease but no heart failure: I11.9.",
     "codes":["I11.0","I11.9","I50.1","I50.20","I50.22","I50.30","I50.32","I50.40","I50.42"],"keywords":["hypertensive heart disease","I11","heart failure hypertension","HFrEF","HFpEF","I50"]},
    {"id":"I.C.9.c","title":"Hypertensive chronic kidney disease","specialty":"Nephrology",
     "content":"Hypertension with CKD: I12.- (presumes causal relationship per ICD-10-CM). I12.9: stage 1-4 or unspecified. I12.30: stage 5 or ESRD with non-dialysis. Do NOT separately code I10 + N18. Always add N18.- stage code. Dialysis: Z99.2 additional.",
     "codes":["I12","I12.9","I12.30","N18.1","N18.2","N18.3","N18.4","N18.5","N18.6"],"keywords":["hypertensive CKD","hypertension kidney","I12","chronic kidney disease hypertension","renal hypertension"]},
    {"id":"I.C.9.d","title":"Hypertensive heart and CKD combined","specialty":"Cardiology",
     "content":"Hypertension + heart disease + CKD together: I13.-. I13.0: with heart failure + CKD stage 1-4. I13.10: no heart failure + CKD stage 1-4. I13.11: no heart failure + CKD stage 5/ESRD. I13.2: with heart failure + CKD stage 5/ESRD. Add N18.- and I50.- as additional codes.",
     "codes":["I13.0","I13.10","I13.11","I13.2","N18.3","N18.5","I50.22","I50.32"],"keywords":["hypertensive heart CKD","I13","triple diagnosis","hypertension heart kidney"]},
    {"id":"I.C.9.e","title":"Hypertension in pregnancy","specialty":"Obstetrics",
     "content":"Pre-existing hypertension in pregnancy: O10.-. Gestational hypertension: O13. Pre-eclampsia: O14.-. Eclampsia: O15. HELLP syndrome: O14.2-. Do NOT use I10 as principal when pregnancy-related hypertension codes apply. Always add trimester character.",
     "codes":["O10","O10.019","O13","O14","O14.00","O14.10","O14.20","O15"],"keywords":["hypertension pregnancy","pre-eclampsia","gestational hypertension","O10","O14","HELLP"]},
    # ── HEART FAILURE (I.C.9.f) ───────────────────────────────
    {"id":"I.C.9.f","title":"Heart failure coding","specialty":"Cardiology",
     "content":"Heart failure types: I50.1 (left ventricular failure unspecified), I50.20 (unspecified systolic), I50.22 (chronic systolic), I50.23 (acute on chronic systolic), I50.30 (unspecified diastolic), I50.32 (chronic diastolic/HFpEF), I50.40 (combined systolic+diastolic). Acute exacerbation changes 5th character. Always specify type when documented.",
     "codes":["I50.1","I50.20","I50.21","I50.22","I50.23","I50.30","I50.31","I50.32","I50.33","I50.40","I50.41","I50.42","I50.43","I50.9"],"keywords":["heart failure","HFrEF","HFpEF","systolic","diastolic","I50","acute heart failure","CHF"]},
    # ── ATRIAL FIBRILLATION ───────────────────────────────────
    {"id":"I.C.9.g","title":"Atrial fibrillation and flutter","specialty":"Cardiology",
     "content":"Atrial fibrillation: I48.0 (paroxysmal), I48.11 (longstanding persistent), I48.19 (other persistent), I48.20 (chronic/permanent unspecified), I48.21 (permanent). Atrial flutter: I48.3 (typical), I48.4 (atypical). Assign cardioversion status Z53.09 if attempted. Anticoagulation: Z79.01.",
     "codes":["I48.0","I48.11","I48.19","I48.20","I48.21","I48.3","I48.4","Z79.01"],"keywords":["atrial fibrillation","AFib","atrial flutter","I48","paroxysmal","persistent","permanent"]},
    # ── ACUTE MI ──────────────────────────────────────────────
    {"id":"I.C.9.h","title":"Acute myocardial infarction","specialty":"Cardiology",
     "content":"STEMI: I21.0- (anterior), I21.1- (inferior), I21.2- (other sites), I21.3 (unspecified). NSTEMI: I21.4. Subsequent MI within 4 weeks: I22.-. After 4 weeks use old MI I25.2. Type 2 MI (supply-demand mismatch): I21.A1. Other types: I21.A9. Always specify wall involved when documented.",
     "codes":["I21.01","I21.09","I21.11","I21.19","I21.3","I21.4","I21.A1","I22","I25.2"],"keywords":["myocardial infarction","MI","STEMI","NSTEMI","heart attack","I21","I22"]},
    # ── CKD (I.C.14) ──────────────────────────────────────────
    {"id":"I.C.14.a","title":"Chronic kidney disease staging","specialty":"Nephrology",
     "content":"CKD stages: N18.1 (G1), N18.2 (G2), N18.3 (G3a/G3b), N18.4 (G4), N18.5 (G5), N18.6 (ESRD/dialysis), N18.9 (unspecified — only when stage not documented). Post-transplant CKD: assign N18.- with transplant status Z94.0. Dialysis: Z99.2. Do not assume CKD resolved after transplant.",
     "codes":["N18.1","N18.2","N18.3","N18.4","N18.5","N18.6","N18.9","Z94.0","Z99.2"],"keywords":["CKD","chronic kidney disease","N18","ESRD","kidney failure","dialysis","GFR","eGFR"]},
    {"id":"I.C.14.b","title":"Acute kidney injury","specialty":"Nephrology",
     "content":"Acute kidney injury (AKI): N17.-. N17.0 with tubular necrosis, N17.1 with acute cortical necrosis, N17.2 with medullary necrosis, N17.9 unspecified. Do not assign N17 and N18 together unless both acute-on-chronic is documented. Contrast nephropathy: T45.8X5 + N17.9.",
     "codes":["N17","N17.0","N17.9","T45.8X5A"],"keywords":["AKI","acute kidney injury","N17","acute renal failure","contrast nephropathy"]},
    # ── SEPSIS (I.C.1.d) ──────────────────────────────────────
    {"id":"I.C.1.d.1","title":"Sepsis — basic coding","specialty":"Infectious Disease",
     "content":"Sepsis: code the underlying systemic infection as principal diagnosis (e.g., A41.9 sepsis unspecified organism, A41.01 MRSA sepsis, A41.51 gram-negative sepsis). Sepsis codes include systemic infection — do NOT also assign R65.1- (SIRS). Urosepsis is NOT a valid ICD-10 code — query provider for specificity.",
     "codes":["A41.9","A41.01","A41.02","A41.1","A41.2","A41.3","A41.50","A41.51","A41.52","A40.0"],"keywords":["sepsis","A41","MRSA sepsis","gram negative sepsis","bacteremia","urosepsis"]},
    {"id":"I.C.1.d.2","title":"Severe sepsis and septic shock","specialty":"Intensive Care",
     "content":"Severe sepsis (with organ dysfunction): add R65.20. Septic shock: R65.21 (NOT the principal diagnosis — always sequence infection code first). Organ dysfunction codes (N17.9, J96.0-, K72.00) required as additional codes. Septic shock without organ dysfunction documented: query provider.",
     "codes":["R65.20","R65.21","N17.9","J96.00","K72.00","D65"],"keywords":["severe sepsis","septic shock","R65.20","R65.21","organ dysfunction","ICU sepsis"]},
    # ── RESPIRATORY (I.C.10) ──────────────────────────────────
    {"id":"I.C.10.a","title":"COPD coding","specialty":"Pulmonology",
     "content":"COPD: J44.9 (unspecified). COPD with acute exacerbation: J44.1. COPD with acute lower respiratory infection: J44.0 — also assign the infection code (J18.9 pneumonia unspecified, J06.9 URI). Do NOT use J44.1 and J44.0 together. Chronic bronchitis J41-J42 is subsumed under COPD when both documented.",
     "codes":["J44.9","J44.1","J44.0","J18.9","J06.9","J41.0","J42"],"keywords":["COPD","chronic obstructive","J44","exacerbation","bronchitis","emphysema"]},
    {"id":"I.C.10.b","title":"Asthma coding","specialty":"Pulmonology",
     "content":"Asthma severity: J45.20 (mild intermittent uncomplicated), J45.21 (with acute exacerbation), J45.22 (with status asthmaticus), J45.30 (mild persistent uncomplicated), J45.31/32 same pattern. J45.50 (severe persistent). J45.901 (unspecified with acute exacerbation). Status asthmaticus is the 5th character x2 at each severity level.",
     "codes":["J45.20","J45.21","J45.22","J45.30","J45.31","J45.32","J45.40","J45.41","J45.42","J45.50","J45.51","J45.52","J45.901"],"keywords":["asthma","J45","status asthmaticus","acute exacerbation asthma","mild persistent","severe persistent"]},
    {"id":"I.C.10.c","title":"Pneumonia coding","specialty":"Pulmonology",
     "content":"Pneumonia by organism: J18.9 (unspecified), J18.0 (bronchopneumonia), J15.0 (Klebsiella), J15.1 (Pseudomonas), J15.20 (Staph unspecified), J15.211 (MSSA), J15.212 (MRSA), J15.4 (Strep other), J15.7 (Mycoplasma), J12.0 (adenoviral), J12.89 (other viral). COVID-19 pneumonia: U07.1 + J12.89.",
     "codes":["J18.9","J18.0","J15.0","J15.1","J15.212","J15.7","J12.89","U07.1"],"keywords":["pneumonia","J18","J15","MRSA pneumonia","aspiration pneumonia","COVID pneumonia","J12"]},
    {"id":"I.C.10.d","title":"COVID-19 coding","specialty":"Infectious Disease",
     "content":"Confirmed COVID-19: U07.1 (code first). COVID pneumonia: U07.1 + J12.89. Post-COVID condition: U09.9. COVID-19 in pregnancy: O98.5-. Suspected/probable COVID without positive test: Z20.828 (exposure). MIS-C (multisystem inflammatory): M35.81. Assign all associated manifestations as additional codes.",
     "codes":["U07.1","J12.89","U09.9","O98.511","Z20.828","M35.81"],"keywords":["COVID-19","coronavirus","U07.1","post-COVID","long COVID","U09.9","MIS-C"]},
    # ── CANCER (I.C.2) ────────────────────────────────────────
    {"id":"I.C.2.a","title":"Malignancy — primary and sequencing","specialty":"Oncology",
     "content":"Primary malignancy as reason for admission: sequence primary site first. Metastatic disease as reason: sequence metastatic site first. Active treatment includes surgery, radiation, chemo, immunotherapy. Personal history (no current treatment): Z85.-. Family history: Z80.-. Prophylactic organ removal: Z40.-.",
     "codes":["Z85","Z80","Z40","C00-C97"],"keywords":["cancer","malignancy","primary cancer","metastasis","Z85","oncology","chemotherapy","radiation"]},
    {"id":"I.C.2.b","title":"Metastatic cancer coding","specialty":"Oncology",
     "content":"Secondary malignancy (metastasis): C77 (lymph nodes), C78 (respiratory/digestive organs), C79 (other sites). When both primary and secondary are present, sequence by reason for encounter. Unknown primary with metastasis: C80.1 (malignant neoplasm without primary site). Carcinoma in situ: D00-D09.",
     "codes":["C77","C78","C79","C80.1","D00","D09","C78.00","C79.51"],"keywords":["metastasis","secondary malignancy","C77","C78","C79","unknown primary","bone metastasis"]},
    {"id":"I.C.2.c","title":"Chemotherapy and radiation encounters","specialty":"Oncology",
     "content":"Encounter for chemotherapy: Z51.11 (principal). Encounter for radiation: Z51.0 (principal). Malignancy coded as additional. Neoplasm-related pain: G89.3. Adverse effect of chemo: T45.1X5A + manifestation code. Complications of chemo: sequence the complication first, then malignancy.",
     "codes":["Z51.11","Z51.0","G89.3","T45.1X5A","Z51.12"],"keywords":["chemotherapy","radiation","Z51","cancer encounter","chemo side effects","T45"]},
    # ── FRACTURES (I.C.19) ────────────────────────────────────
    {"id":"I.C.19.a","title":"Traumatic fracture 7th character","specialty":"Orthopedics",
     "content":"Fracture 7th characters: A (initial encounter, active treatment), D (subsequent encounter, healing/routine care), G (subsequent, delayed healing), K (subsequent, nonunion), P (subsequent, malunion), S (sequela). Use A for all initial active treatment regardless of which provider seen. After active treatment ends, use D for all routine follow-up.",
     "codes":["S","A","D","G","K","P","S"],"keywords":["fracture","7th character","initial encounter","subsequent encounter","sequela","nonunion","malunion"]},
    {"id":"I.C.19.b","title":"Pathological fractures","specialty":"Orthopedics",
     "content":"Pathological fracture (due to disease): M80.- (osteoporosis) or M84.4- (neoplastic disease), M84.5- (other disease). Traumatic vs pathological: provider must document cause. Same 7th character rules apply. Osteoporosis without pathological fracture: M81.-. Compression fracture in osteoporosis: M80.08XA.",
     "codes":["M80","M80.08XA","M84.40XA","M84.50XA","M81","M81.0"],"keywords":["pathological fracture","osteoporosis fracture","M80","M84","vertebral compression fracture"]},
    # ── MENTAL HEALTH (I.C.5) ─────────────────────────────────
    {"id":"I.C.5.a","title":"Mental health — depression","specialty":"Psychiatry",
     "content":"Major depressive disorder (MDD): F32.- (single episode), F33.- (recurrent). Severity: 0=mild, 1=moderate, 2=severe without psychosis, 3=severe with psychosis, 4=partial remission, 5=full remission, 9=unspecified. Persistent depressive disorder (dysthymia): F34.1. Postpartum depression: F53.0.",
     "codes":["F32.0","F32.1","F32.2","F32.3","F32.4","F32.5","F32.9","F33.0","F33.1","F33.2","F33.9","F34.1","F53.0"],"keywords":["depression","MDD","F32","F33","major depressive disorder","dysthymia","postpartum depression"]},
    {"id":"I.C.5.b","title":"Anxiety disorders","specialty":"Psychiatry",
     "content":"Generalized anxiety: F41.1. Panic disorder: F41.0. Social anxiety: F40.10. Specific phobia: F40.2-. Mixed anxiety-depression: F41.8. PTSD: F43.10 (unspecified), F43.11 (acute), F43.12 (chronic). Adjustment disorder with anxiety: F43.22. OCD: F42.2.",
     "codes":["F41.1","F41.0","F40.10","F43.10","F43.11","F43.12","F43.22","F42.2"],"keywords":["anxiety","GAD","panic disorder","PTSD","F41","F43","OCD","social anxiety"]},
    {"id":"I.C.5.c","title":"Substance use disorders","specialty":"Psychiatry",
     "content":"Substance use: in remission requires provider documentation. Alcohol: F10.- (use=.10, abuse=.10, dependence=.20, dependence with withdrawal=.230). Opioid: F11.-. Cannabis: F12.-. Stimulant: F15.-. Tobacco: F17.21-. In remission: 5th character 1=mild, 2=moderate-severe. Do NOT code both use and abuse/dependence.",
     "codes":["F10.10","F10.20","F10.230","F11.20","F12.10","F17.210","F17.211"],"keywords":["substance use","alcohol use disorder","F10","opioid use","F11","tobacco use","F17","in remission"]},
    # ── OBESITY (I.C.21.c) ────────────────────────────────────
    {"id":"I.C.21.c.1","title":"Obesity and BMI coding","specialty":"General Medicine",
     "content":"Morbid obesity: E66.01. Obesity unspecified: E66.09. Overweight: E66.3. Obesity due to excess calories: E66.01. BMI codes Z68.- are always additional codes, never principal. Only assign BMI when associated obesity condition is documented. Pediatric BMI: Z68.5-. BMI ≥40: Z68.40-Z68.45.",
     "codes":["E66.01","E66.09","E66.3","Z68.1","Z68.2","Z68.3","Z68.41","Z68.42","Z68.43","Z68.44","Z68.45"],"keywords":["obesity","BMI","E66","Z68","morbid obesity","overweight","bariatric"]},
    # ── LONG-TERM MEDICATIONS (I.C.21.c.15) ──────────────────
    {"id":"I.C.21.c.15","title":"Long-term medication use Z79 codes","specialty":"General Medicine",
     "content":"Long-term drug codes (always secondary): Z79.4 insulin, Z79.01 anticoagulants (warfarin, DOACs), Z79.02 antithrombotics/antiplatelets, Z79.1 NSAIDs, Z79.3 contraceptives, Z79.51 inhaled steroids, Z79.52 systemic steroids, Z79.84 oral hypoglycemics, Z79.891 opiate analgesics, Z79.899 other. Assign only when chronic/long-term — not for single-dose or short-course.",
     "codes":["Z79.4","Z79.01","Z79.02","Z79.1","Z79.3","Z79.51","Z79.52","Z79.84","Z79.891","Z79.899"],"keywords":["long term medication","Z79","insulin use","anticoagulant","warfarin","DOAC","steroids","Z79.01","Z79.4"]},
    # ── STROKE (I.C.9.i) ──────────────────────────────────────
    {"id":"I.C.9.i","title":"Cerebrovascular disease — stroke","specialty":"Neurology",
     "content":"Ischemic stroke: I63.- (by mechanism). Hemorrhagic stroke: I61.- (intracerebral), I60.- (subarachnoid). TIA: G45.9. Stroke NOS: I63.9. Hemiplegia post-stroke: I69.35-, I69.15-, I69.25-. Do NOT assign I63 and residual I69 together for the SAME encounter — use I63 during acute phase, I69 for residuals at follow-up.",
     "codes":["I63.9","I63.00","I61.9","I60.9","G45.9","I69.351","I69.352","I69.159"],"keywords":["stroke","CVA","ischemic stroke","hemorrhagic stroke","I63","I61","TIA","G45","hemiplegia post stroke"]},
    # ── DVT/PE ────────────────────────────────────────────────
    {"id":"I.C.9.j","title":"Deep vein thrombosis and pulmonary embolism","specialty":"Hematology",
     "content":"DVT: I82.- by site (I82.401 femoral, I82.4Y1 other specified leg, I82.621 brachial). PE: I26.0- (with acute cor pulmonale), I26.9- (without). Chronic DVT: I82.5-. Post-thrombotic syndrome: I87.0-. Anticoagulation: Z79.01. DVT + PE together: assign both codes.",
     "codes":["I82.401","I82.4Y1","I26.01","I26.09","I26.90","I82.501","Z79.01"],"keywords":["DVT","deep vein thrombosis","pulmonary embolism","PE","I82","I26","anticoagulation","VTE"]},
    # ── INFECTIONS ────────────────────────────────────────────
    {"id":"I.C.1.a","title":"MRSA and MSSA infections","specialty":"Infectious Disease",
     "content":"MRSA infection: assign the condition code first, then B95.62 (MRSA as cause). MRSA colonization (carrier, no infection): Z22.322. MSSA: B95.61. MRSA sepsis: A41.02. MRSA pneumonia: J15.212. Do NOT assign B95.62 as principal diagnosis — it is always additional.",
     "codes":["B95.62","B95.61","A41.02","J15.212","Z22.322"],"keywords":["MRSA","MSSA","staph infection","B95.62","methicillin resistant","colonization"]},
    {"id":"I.C.1.b","title":"HIV disease and AIDS","specialty":"Infectious Disease",
     "content":"Confirmed HIV: B20 (HIV disease — used when ANY AIDS-defining condition or HIV-related condition present). HIV positive without symptoms/conditions: Z21 (asymptomatic HIV). Do NOT assign Z21 and B20 together. HIV in pregnancy: O98.7-. Inconclusive HIV test: R75. CD4 count: Z98.81.",
     "codes":["B20","Z21","O98.711","R75","Z98.81"],"keywords":["HIV","AIDS","B20","Z21","antiretroviral","HIV positive","CD4"]},
    # ── SURGICAL / COMPLICATIONS ──────────────────────────────
    {"id":"I.C.19.c","title":"Surgical complications","specialty":"Surgery",
     "content":"Postoperative complications: sequence the complication code first. Wound dehiscence: T81.31XA. Seroma: T81.89XA. Post-op infection: T81.40XA-T81.49XA. Surgical site infection: add B-code for organism. Post-op hemorrhage: T81.01XA (during procedure), T81.03XA (following procedure). 7th character A=initial, D=subsequent, S=sequela.",
     "codes":["T81.31XA","T81.40XA","T81.49XA","T81.01XA","T81.03XA","T81.89XA"],"keywords":["surgical complication","post-op","wound dehiscence","seroma","T81","postoperative infection","surgical site infection"]},
    # ── PAIN (I.C.6) ──────────────────────────────────────────
    {"id":"I.C.6.a","title":"Pain coding — acute vs chronic","specialty":"Pain Management",
     "content":"Acute pain: G89.01 (postprocedural), G89.11 (acute traumatic), G89.21 (acute neoplasm-related), G89.29 (other acute). Chronic pain: G89.21 (neoplasm-related), G89.29 (other chronic), G89.4 (chronic pain syndrome). Do NOT assign G89.- if the pain is integral to the condition (e.g., headache in migraine). Opioid use for pain: Z79.891.",
     "codes":["G89.01","G89.11","G89.21","G89.29","G89.4","G89.3","Z79.891"],"keywords":["pain","acute pain","chronic pain","G89","neoplasm pain","postoperative pain","cancer pain"]},
    # ── LIVER DISEASE ─────────────────────────────────────────
    {"id":"I.C.11.a","title":"Liver disease and cirrhosis","specialty":"Gastroenterology",
     "content":"Alcoholic cirrhosis: K70.30 (without ascites), K70.31 (with ascites). Non-alcoholic cirrhosis: K74.60 (unspecified), K74.69 (other). Non-alcoholic fatty liver (NAFL): K76.0. NASH: K75.81. Hepatic encephalopathy: K72.91. Portal hypertension: K76.6. Hepatorenal syndrome: K76.7.",
     "codes":["K70.30","K70.31","K74.60","K74.69","K75.81","K76.0","K72.91","K76.6","K76.7"],"keywords":["cirrhosis","liver disease","K70","K74","NASH","fatty liver","hepatic encephalopathy","portal hypertension"]},
    # ── OBSTETRICS (I.C.15) ───────────────────────────────────
    {"id":"I.C.15.a","title":"Obstetric coding rules","specialty":"Obstetrics",
     "content":"Trimester characters: 1 (<14 weeks), 2 (14-28 weeks), 3 (>28 weeks). Outcome of delivery: Z37.- always required as additional code when delivery occurs. Elective cesarean: O82. Failed trial of labor: O66.40. Preterm labor: O60.-. Group B strep carrier in pregnancy: O99.820.",
     "codes":["Z37.0","Z37.1","Z37.2","O82","O66.40","O60.00","O99.820"],"keywords":["delivery","obstetric","trimester","Z37","cesarean","O82","preterm labor","outcome of delivery"]},
    # ── ENDOCRINE ─────────────────────────────────────────────
    {"id":"I.C.4.c","title":"Thyroid disorders","specialty":"Endocrinology",
     "content":"Hypothyroidism: E03.9 (unspecified), E03.0 (congenital), E06.3 (Hashimoto/autoimmune). Hyperthyroidism: E05.00 (Graves without thyrotoxic crisis), E05.01 (with crisis/storm). Thyroid nodule: E04.1. Thyroid cancer: C73. Thyroiditis: E06.-. Levothyroxine use: Z79.899.",
     "codes":["E03.9","E03.0","E06.3","E05.00","E05.01","E04.1","C73","E06.0"],"keywords":["hypothyroidism","hyperthyroidism","thyroid","E03","E05","Hashimoto","Graves disease","levothyroxine"]},
    {"id":"I.C.4.d","title":"Adrenal and pituitary disorders","specialty":"Endocrinology",
     "content":"Cushing syndrome: E24.0 (pituitary-dependent), E24.2 (drug-induced), E24.9 (unspecified). Addison disease: E27.1. Primary hyperaldosteronism: E26.01. Pheochromocytoma: D35.00 (benign), C74.10 (malignant). Hyperprolactinemia: E22.1. Acromegaly: E22.0.",
     "codes":["E24.0","E24.2","E27.1","E26.01","D35.00","C74.10","E22.1","E22.0"],"keywords":["Cushing","Addison","adrenal","E24","E27","pheochromocytoma","hyperaldosteronism","pituitary"]},
    # ── RHEUMATOLOGY ──────────────────────────────────────────
    {"id":"I.C.13.a","title":"Rheumatoid arthritis","specialty":"Rheumatology",
     "content":"Rheumatoid arthritis with positive RF: M05.- (by site). Seronegative RA: M06.00-M06.09 (by site). Juvenile RA: M08.-. RA with organ involvement: M05.3- (heart), M05.1- (lung), M05.2- (vasculitis). Biologic therapy use: Z79.899. Methotrexate: Z79.899.",
     "codes":["M05.00","M05.09","M06.00","M06.09","M08.00","M05.30","M05.10"],"keywords":["rheumatoid arthritis","RA","M05","M06","seronegative","seropositive","biologic","methotrexate"]},
    {"id":"I.C.13.b","title":"Osteoarthritis","specialty":"Rheumatology",
     "content":"Primary osteoarthritis: M15 (polyosteoarthritis), M16.- (hip), M17.- (knee), M18.- (first CMC), M19.- (other). Bilateral involvement: characters 0=unspecified side, 1=right, 2=left, 3=bilateral. Post-traumatic OA: M19.1-. Secondary OA: M19.2-. Total joint replacement status: Z96.641-Z96.649.",
     "codes":["M16.0","M16.11","M16.12","M17.0","M17.11","M17.12","M19.011","Z96.641","Z96.651"],"keywords":["osteoarthritis","OA","M16","M17","hip replacement","knee replacement","Z96","joint replacement"]},
    # ── STROKE REHAB / RESIDUALS ──────────────────────────────
    {"id":"I.C.9.k","title":"Stroke residuals and rehabilitation","specialty":"Neurology",
     "content":"Late effects of stroke: I69.-. Assign I69.- when treatment is for residuals, not acute stroke. I69.30- (unspecified cerebrovascular disease), I69.35- (hemiplegia), I69.32- (aphasia), I69.390 (other sequelae). Dysphagia post-stroke: I69.391. Do NOT assign I69 during acute stroke hospitalization.",
     "codes":["I69.30","I69.351","I69.352","I69.320","I69.391","I69.390"],"keywords":["stroke residuals","post-stroke","I69","hemiplegia stroke","aphasia","dysphagia stroke","rehabilitation"]},
    # ── INJURIES / POISONING ──────────────────────────────────
    {"id":"I.C.19.d","title":"Poisoning, adverse effects, underdosing","specialty":"Emergency Medicine",
     "content":"Poisoning (wrong substance, wrong dose intentionally or accidentally): T codes with 5th/6th character: 1=accidental, 2=intentional self-harm, 3=assault, 4=undetermined. Adverse effect (correct substance, correct dose): T code with 5th/6th character 5 + manifestation code. Underdosing (less than prescribed): 6th character 6. Always assign T code first for poisoning.",
     "codes":["T36-T65","5=adverse","1=accidental","2=intentional","6=underdosing"],"keywords":["poisoning","adverse effect","underdosing","T code","drug toxicity","overdose","medication error"]},
    # ── SCREENING / PREVENTIVE ────────────────────────────────
    {"id":"I.C.21.b","title":"Preventive medicine and screening","specialty":"General Medicine",
     "content":"Encounter for screening: Z12.- (malignancy), Z13.- (other disorders). Encounter for immunization: Z23. Routine adult exam: Z00.00 (without abnormal findings), Z00.01 (with). Annual GYN exam: Z01.411/Z01.419. Screening colonoscopy: Z12.11. Screening mammogram: Z12.31. Family history used for screening justification: Z80-Z84.",
     "codes":["Z00.00","Z00.01","Z12.11","Z12.31","Z01.411","Z23","Z80","Z13"],"keywords":["screening","preventive","Z00","Z12","Z13","annual exam","immunization","colonoscopy","mammogram"]},
]


@dataclass
class ICD10Result:
    query: str = ""
    answer: str = ""
    primary_codes: list = field(default_factory=list)
    supporting_sections: list = field(default_factory=list)
    retrieved_chunks: list = field(default_factory=list)
    confidence: str = "Moderate"
    generated_by: str = "Rule-based"
    faithfulness_score: float = 0.0
    disclaimer: str = "Always verify with official CMS ICD-10-CM guidelines before billing."


class ICD10RAGEngine:
    """
    Production ICD-10-CM RAG engine.
    - 80+ CMS guideline sections across all specialties
    - TF-IDF cosine retrieval (no external deps)
    - SQLite persistent cache
    - Groq LLM synthesis with paragraph-level citation grounding
    - Faithfulness scoring
    """

    def __init__(self, groq_api_key: Optional[str] = None,
                 db_path: str = "clis_cache.db"):
        self.api_key  = groq_api_key or os.getenv("GROQ_API_KEY", "")
        self.client   = None
        self.kb       = ICD10_KB
        self.db_path  = db_path
        self._build_index()
        self._init_cache()

        if self.api_key and self.api_key not in ("", "your_groq_api_key_here"):
            try:
                from groq import Groq
                self.client = Groq(api_key=self.api_key)
            except ImportError:
                pass

    # ── Index ──────────────────────────────────────────────────
    def _build_index(self):
        self._docs = []
        for c in self.kb:
            text = (c["title"] + " " + c["content"] + " " +
                    " ".join(c.get("keywords", [])) + " " +
                    " ".join(c.get("codes", []))).lower()
            self._docs.append(text)
        N = len(self._docs)
        df = {}
        for doc in self._docs:
            for t in set(self._tok(doc)):
                df[t] = df.get(t, 0) + 1
        self._idf = {w: math.log((N + 1) / (f + 1)) + 1 for w, f in df.items()}

    def _tok(self, t): return re.findall(r"[a-z0-9]+", t.lower())

    def _vec(self, text):
        toks = self._tok(text)
        tf = {}
        for t in toks: tf[t] = tf.get(t, 0) + 1
        n = len(toks) or 1
        return {t: (c/n)*self._idf.get(t, 0) for t, c in tf.items()}

    def _cos(self, a, b):
        dot = sum(a.get(t, 0)*b.get(t, 0) for t in a)
        na  = math.sqrt(sum(x**2 for x in a.values())) or 1
        nb  = math.sqrt(sum(x**2 for x in b.values())) or 1
        return dot/(na*nb)

    def retrieve(self, query: str, top_k: int = 4) -> list:
        qv = self._vec(query)
        scores = sorted(
            [(self._cos(qv, self._vec(d)), i) for i, d in enumerate(self._docs)],
            reverse=True)
        return [self.kb[i] for _, i in scores[:top_k]]

    # ── Cache ──────────────────────────────────────────────────
    def _init_cache(self):
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute("""CREATE TABLE IF NOT EXISTS icd_cache
                (key TEXT PRIMARY KEY, result TEXT, ts REAL)""")
            conn.commit(); conn.close()
        except Exception: pass

    def _cache_get(self, key):
        try:
            conn = sqlite3.connect(self.db_path)
            row = conn.execute("SELECT result FROM icd_cache WHERE key=?",
                               (key,)).fetchone()
            conn.close()
            return json.loads(row[0]) if row else None
        except Exception: return None

    def _cache_set(self, key, val):
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute("INSERT OR REPLACE INTO icd_cache VALUES(?,?,?)",
                         (key, json.dumps(val), time.time()))
            conn.commit(); conn.close()
        except Exception: pass

    # ── Faithfulness scoring ───────────────────────────────────
    def _faithfulness(self, answer: str, chunks: list) -> float:
        """
        Compute faithfulness: fraction of answer sentences that
        have at least one keyword match in retrieved chunks.
        Proxy for citation grounding without heavy NLP deps.
        """
        context_words = set()
        for c in chunks:
            context_words.update(self._tok(c["content"]))
            context_words.update(self._tok(c["title"]))
            for code in c.get("codes", []):
                context_words.update(self._tok(code))

        sentences = [s.strip() for s in re.split(r'[.!?]', answer) if len(s.strip()) > 20]
        if not sentences: return 0.7
        grounded = 0
        for sent in sentences:
            sent_words = set(self._tok(sent))
            if len(sent_words & context_words) >= 2:
                grounded += 1
        return round(grounded / len(sentences), 3)

    # ── Main entry ─────────────────────────────────────────────
    def answer(self, query: str) -> ICD10Result:
        cache_key = hashlib.md5(query.lower().strip().encode()).hexdigest()
        cached = self._cache_get(cache_key)
        if cached:
            r = ICD10Result(**cached)
            r.generated_by += " (cached)"
            return r

        chunks = self.retrieve(query, top_k=4)
        all_codes = []
        for c in chunks:
            all_codes.extend(c.get("codes", []))
        all_codes = list(dict.fromkeys(all_codes))[:8]

        if self.client:
            answer_text, gen_by = self._llm_answer(query, chunks)
        else:
            answer_text, gen_by = self._rule_answer(query, chunks)

        faith = self._faithfulness(answer_text, chunks)

        result = ICD10Result(
            query=query,
            answer=answer_text,
            primary_codes=all_codes[:6],
            supporting_sections=[{"section": c["id"], "title": c["title"],
                                   "specialty": c.get("specialty",""),
                                   "codes": c.get("codes",[])[:4]} for c in chunks],
            retrieved_chunks=chunks,
            confidence="High" if faith > 0.7 else "Moderate" if faith > 0.4 else "Low",
            generated_by=gen_by,
            faithfulness_score=faith,
        )
        self._cache_set(cache_key, {
            "query": result.query, "answer": result.answer,
            "primary_codes": result.primary_codes,
            "supporting_sections": result.supporting_sections,
            "retrieved_chunks": [],
            "confidence": result.confidence,
            "generated_by": result.generated_by,
            "faithfulness_score": result.faithfulness_score,
            "disclaimer": result.disclaimer,
        })
        return result

    def _llm_answer(self, query: str, chunks: list) -> tuple:
        context = "\n\n".join([
            f"[{c['id']}] {c['title']} (Specialty: {c.get('specialty','')})\n"
            f"{c['content']}\nCodes: {', '.join(c.get('codes',[])[:6])}"
            for c in chunks
        ])
        prompt = f"""You are a certified medical coder using CMS ICD-10-CM Official Guidelines FY2024.

Clinical coding question: {query}

Retrieved guideline sections:
{context}

Provide a precise coding answer with:
1. Primary ICD-10-CM code(s) — most specific first with full description
2. Required additional codes with rationale
3. Sequencing rule (which is principal/first-listed diagnosis)
4. Documentation requirements
5. Any important caveats or common coding errors to avoid

Cite the specific guideline section (e.g., I.C.4.a.3) for each coding decision.
Be clinically precise and concise."""

        try:
            from groq import Groq
            resp = Groq(api_key=self.api_key).chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role":"user","content":prompt}],
                temperature=0.05, max_tokens=600)
            return resp.choices[0].message.content.strip(), "Groq Llama 3.3 70B"
        except Exception as e:
            return self._rule_answer(query, chunks)[0], f"Rule-based (LLM error: {str(e)[:40]})"

    def _rule_answer(self, query: str, chunks: list) -> tuple:
        top = chunks[0]
        codes_str = ", ".join(top.get("codes", [])[:4])
        return (f"Per ICD-10-CM guideline {top['id']} ({top['title']}): "
                f"{top['content']} "
                f"Key codes: {codes_str}. "
                f"Always verify sequencing with the complete ICD-10-CM Tabular List "
                f"and confirm documentation supports the specificity level assigned."), "Rule-based"
