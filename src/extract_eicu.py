"""eICU-CRD 2.0 -> same hour-12 landmark cohort and 22-channel hourly long table as MIMIC-IV.

Offsets are minutes from unit admission. Death time = hospital discharge offset when status 'Expired'.
Outputs (data/): eicu_cohort.parquet, eicu_long.parquet
"""
import os
import time
import duckdb

E = os.environ.get("EICU_DIR", "/A.I_DATA/jbnu/jaehong/중증도/eicu-collaborative-research-database-2.0")
OUT = os.path.join(os.environ.get("PF_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")

con = duckdb.connect()
con.execute("SET threads=16; SET memory_limit='48GB'; SET preserve_insertion_order=false; SET enable_progress_bar=false")
t0 = time.time()


def log(msg):
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


def rd(name):
    return f"read_csv('{E}/{name}.csv.gz', header=true, quote='\"', strict_mode=false, sample_size=200000)"


con.execute(f"""
CREATE TABLE pat AS
SELECT patientunitstayid AS stay_id, uniquepid, patienthealthsystemstayid, hospitalid, gender,
       CASE WHEN age = '> 89' THEN 90 ELSE try_cast(age AS INT) END AS age,
       hospitaldischargeyear, hospitaladmitoffset, unitvisitnumber,
       unitdischargeoffset, hospitaldischargeoffset, hospitaldischargestatus,
       coalesce(admissionweight, dischargeweight) AS weight
FROM {rd('patient')}
""")
n0 = con.sql("select count(*) from pat").fetchone()[0]
con.execute("CREATE TABLE s1 AS SELECT * FROM pat WHERE age BETWEEN 18 AND 80")
n1 = con.sql("select count(*) from s1").fetchone()[0]
con.execute("""CREATE TABLE s2 AS SELECT * FROM s1 QUALIFY row_number() OVER (
  PARTITION BY uniquepid ORDER BY hospitaldischargeyear, patienthealthsystemstayid, unitvisitnumber, stay_id) = 1""")
n2 = con.sql("select count(*) from s2").fetchone()[0]
con.execute("""CREATE TABLE s3 AS SELECT * FROM s2 WHERE unitdischargeoffset >= 720
  AND NOT (hospitaldischargestatus = 'Expired' AND hospitaldischargeoffset <= 720)""")
n3 = con.sql("select count(*) from s3").fetchone()[0]
con.execute("""CREATE TABLE cohort AS SELECT *,
  CASE WHEN hospitaldischargestatus = 'Expired' THEN hospitaldischargeoffset/60.0 END AS t_death_h,
  hospitaldischargeoffset/60.0 AS t_disch_h,
  (hospitaldischargestatus = 'Expired')::INT AS hospital_expire_flag
FROM s3 WHERE hospitaldischargestatus IN ('Alive', 'Expired') AND hospitaldischargeoffset IS NOT NULL""")
n4 = con.sql("select count(*) from cohort").fetchone()[0]
log(f"cohort funnel: all={n0} age={n1} first={n2} alive_in_icu_h12={n3} known_status={n4}")

W = "o >= 0 AND o < 720"   # minutes window [0, 12h)

con.execute(f"""
CREATE TABLE vp AS
SELECT v.patientunitstayid stay_id, v.observationoffset o, heartrate, respiration, sao2, temperature,
       systemicsystolic, systemicdiastolic
FROM {rd('vitalPeriodic')} v SEMI JOIN cohort c ON c.stay_id = v.patientunitstayid
WHERE v.observationoffset >= 0 AND v.observationoffset < 720""")
log("vitalPeriodic done")
con.execute(f"""
CREATE TABLE va AS
SELECT v.patientunitstayid stay_id, v.observationoffset o, noninvasivesystolic, noninvasivediastolic
FROM {rd('vitalAperiodic')} v SEMI JOIN cohort c ON c.stay_id = v.patientunitstayid
WHERE v.observationoffset >= 0 AND v.observationoffset < 720""")
con.execute(f"""
CREATE TABLE nc AS
SELECT n.patientunitstayid stay_id, n.nursingchartoffset o, n.nursingchartcelltypevalname nm,
       try_cast(n.nursingchartvalue AS DOUBLE) val
FROM {rd('nurseCharting')} n SEMI JOIN cohort c ON c.stay_id = n.patientunitstayid
WHERE n.nursingchartoffset >= 0 AND n.nursingchartoffset < 720
  AND n.nursingchartcelltypevalname IN ('GCS Total', 'Temperature (C)', 'Heart Rate', 'Respiratory Rate',
                                        'O2 Saturation', 'Non-Invasive BP Systolic', 'Non-Invasive BP Diastolic',
                                        'Invasive BP Systolic', 'Invasive BP Diastolic')""")
log("nurseCharting done")
con.execute(f"""
CREATE TABLE lb AS
SELECT l.patientunitstayid stay_id, l.labresultoffset o,
       CASE l.labname WHEN 'BUN' THEN 'bun' WHEN 'lactate' THEN 'lactate' WHEN 'WBC x 1000' THEN 'wbc'
            WHEN 'Hgb' THEN 'hb' WHEN 'sodium' THEN 'na' WHEN 'potassium' THEN 'k' WHEN 'chloride' THEN 'cl'
            WHEN 'pH' THEN 'ph' WHEN 'Base Excess' THEN 'be' WHEN 'anion gap' THEN 'ag' WHEN 'FiO2' THEN 'fio2' END var,
       l.labresult::DOUBLE val
FROM {rd('lab')} l SEMI JOIN cohort c ON c.stay_id = l.patientunitstayid
WHERE l.labresultoffset >= 0 AND l.labresultoffset < 720 AND l.labresult IS NOT NULL
  AND l.labname IN ('BUN','lactate','WBC x 1000','Hgb','sodium','potassium','chloride','pH','Base Excess','anion gap','FiO2')""")
log("lab done")
con.execute(f"""
CREATE TABLE rc AS
SELECT r.patientunitstayid stay_id, r.respchartoffset o, r.respchartvaluelabel lbl,
       try_cast(replace(r.respchartvalue, '%', '') AS DOUBLE) val
FROM {rd('respiratoryCharting')} r SEMI JOIN cohort c ON c.stay_id = r.patientunitstayid
WHERE r.respchartoffset >= 0 AND r.respchartoffset < 720
  AND r.respchartvaluelabel IN ('FiO2', 'FIO2 (%)', 'Set Fraction of Inspired Oxygen (FIO2)', 'Vent Rate',
                                'Mechanical Ventilator Mode', 'Ventilator Support Mode',
                                'Adult Con Pt/Vent MinuteVentil', 'Adult Con Pt/Vent InspiratorTV')""")
con.execute(f"""
CREATE TABLE tr AS
SELECT t.patientunitstayid stay_id, t.treatmentoffset o
FROM {rd('treatment')} t SEMI JOIN cohort c ON c.stay_id = t.patientunitstayid
WHERE t.treatmentoffset >= 0 AND t.treatmentoffset < 720
  AND lower(t.treatmentstring) LIKE '%mechanical ventilation%' AND lower(t.treatmentstring) NOT LIKE '%non-invasive%'""")
con.execute(f"""
CREATE TABLE inf AS
SELECT i.patientunitstayid stay_id, i.infusionoffset o, lower(i.drugname) dn,
       try_cast(i.drugrate AS DOUBLE) rate, try_cast(i.patientweight AS DOUBLE) pw
FROM {rd('infusionDrug')} i SEMI JOIN cohort c ON c.stay_id = i.patientunitstayid
WHERE i.infusionoffset >= 0 AND i.infusionoffset < 720
  AND (lower(i.drugname) LIKE '%norepinephrine%' OR lower(i.drugname) LIKE '%levophed%' OR lower(i.drugname) LIKE '%vasopressin%')""")
con.execute(f"""
CREATE TABLE io AS
SELECT x.patientunitstayid stay_id, x.intakeoutputoffset o, x.cellvaluenumeric::DOUBLE val
FROM {rd('intakeOutput')} x SEMI JOIN cohort c ON c.stay_id = x.patientunitstayid
WHERE x.intakeoutputoffset >= 0 AND x.intakeoutputoffset < 720
  AND lower(x.cellpath) LIKE '%output%' AND lower(x.celllabel) LIKE '%urine%'
  AND lower(x.celllabel) NOT SIMILAR TO '.*(count|occurrence|stool|incontinen).*'""")
log("other tables done")

con.execute("""
CREATE TABLE long AS
WITH v AS (
  SELECT stay_id, o, 'hr' var, heartrate val FROM vp UNION ALL
  SELECT stay_id, o, 'rr', respiration FROM vp UNION ALL
  SELECT stay_id, o, 'spo2', sao2 FROM vp UNION ALL
  SELECT stay_id, o, 'temp', temperature FROM vp UNION ALL
  SELECT stay_id, o, 'sbp', systemicsystolic FROM vp UNION ALL
  SELECT stay_id, o, 'dbp', systemicdiastolic FROM vp UNION ALL
  SELECT stay_id, o, 'sbp', noninvasivesystolic FROM va UNION ALL
  SELECT stay_id, o, 'dbp', noninvasivediastolic FROM va UNION ALL
  SELECT stay_id, o, CASE nm WHEN 'GCS Total' THEN 'gcs' WHEN 'Temperature (C)' THEN 'temp'
                             WHEN 'Heart Rate' THEN 'hr' WHEN 'Respiratory Rate' THEN 'rr'
                             WHEN 'O2 Saturation' THEN 'spo2'
                             WHEN 'Non-Invasive BP Systolic' THEN 'sbp' WHEN 'Invasive BP Systolic' THEN 'sbp'
                             ELSE 'dbp' END, val FROM nc UNION ALL
  SELECT stay_id, o, var, val FROM lb UNION ALL
  SELECT stay_id, o, 'fio2', val FROM rc WHERE lbl IN ('FiO2','FIO2 (%)','Set Fraction of Inspired Oxygen (FIO2)')
),
v2 AS (
  SELECT stay_id, floor(o/60)::INT AS hour, var,
         arg_max(CASE WHEN var = 'fio2' AND val <= 1.0 THEN val*100 ELSE val END, o) AS value
  FROM v WHERE val IS NOT NULL GROUP BY ALL
),
vent AS (
  SELECT DISTINCT stay_id, floor(o/60)::INT AS hour, 'vent' var, 1.0 AS value FROM (
    SELECT stay_id, o FROM rc WHERE lbl NOT IN ('FiO2','FIO2 (%)','Set Fraction of Inspired Oxygen (FIO2)')
    UNION ALL SELECT stay_id, o FROM tr)
),
ne AS (
  SELECT i.stay_id, floor(i.o/60)::INT AS hour, 'ne' var,
         max(CASE WHEN dn LIKE '%mcg/kg/min%' THEN rate
                  WHEN dn LIKE '%mcg/min%' AND coalesce(pw, c.weight) > 0 THEN rate / coalesce(pw, c.weight) END) AS value
  FROM inf i JOIN cohort c USING (stay_id)
  WHERE (dn LIKE '%norepinephrine%' OR dn LIKE '%levophed%') AND rate IS NOT NULL GROUP BY ALL
),
vaso AS (
  SELECT DISTINCT stay_id, floor(o/60)::INT AS hour, 'vaso' var, 1.0 AS value FROM inf
  WHERE dn LIKE '%vasopressin%' AND coalesce(rate, 1) > 0
),
uo AS (
  SELECT stay_id, floor(o/60)::INT AS hour, 'uo' var, sum(val) AS value FROM io WHERE val IS NOT NULL AND val >= 0 GROUP BY ALL
)
SELECT * FROM v2 UNION ALL SELECT * FROM vent UNION ALL SELECT * FROM ne UNION ALL SELECT * FROM vaso UNION ALL SELECT * FROM uo
""")
con.execute(f"COPY (SELECT * FROM long WHERE hour BETWEEN 0 AND 11 AND value IS NOT NULL) TO '{OUT}/eicu_long.parquet' (FORMAT parquet)")
con.execute(f"""COPY (SELECT stay_id, uniquepid AS subject_id, hospitalid, age, gender, hospital_expire_flag,
                 t_death_h, t_disch_h FROM cohort) TO '{OUT}/eicu_cohort.parquet' (FORMAT parquet)""")
log(con.sql("SELECT var, count(*) n, count(DISTINCT stay_id) stays FROM long GROUP BY var ORDER BY var").fetchall())
log("done")
