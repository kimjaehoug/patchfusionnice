"""MIMIC-IV 3.1 -> hour-12 landmark cohort + first-12-hour hourly long table.

Follows the manuscript protocol (Supplementary Methods):
  adults 18-80, first eligible ICU stay per patient, alive and still in ICU at hour 12,
  unknown in-hospital death time excluded; endpoints (12,24], (12,48], (12,168], in-hospital.
Outputs (data/):
  mimic_cohort.parquet  one row per stay with labels and event times
  mimic_long.parquet    (stay_id, hour, var, value) for hours 0..11
"""
import os
import time
import duckdb

M = os.environ.get("MIMIC_DIR", "/A.I_DATA/jbnu/jaehong/중증도/mimic-iv-3.1")   # PhysioNet MIMIC-IV 3.1 root
OUT = os.path.join(os.environ.get("PF_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")
os.makedirs(OUT, exist_ok=True)

con = duckdb.connect()
con.execute("SET threads=16; SET memory_limit='48GB'; SET preserve_insertion_order=false;")
t0 = time.time()


def log(msg):
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


# ---------------------------------------------------------------- cohort
con.execute(f"""
CREATE TABLE icu AS
SELECT i.subject_id, i.hadm_id, i.stay_id, i.intime::TIMESTAMP intime, i.outtime::TIMESTAMP outtime,
       a.admittime::TIMESTAMP admittime, a.dischtime::TIMESTAMP dischtime, a.deathtime::TIMESTAMP deathtime,
       a.hospital_expire_flag, p.gender,
       p.anchor_age + (year(i.intime::TIMESTAMP) - p.anchor_year) AS age
FROM read_csv_auto('{M}/icu/icustays.csv.gz') i
JOIN read_csv_auto('{M}/hosp/admissions.csv') a USING (hadm_id)
JOIN read_csv_auto('{M}/hosp/patients.csv.gz') p ON p.subject_id = i.subject_id
""")
n0 = con.sql("select count(*) from icu").fetchone()[0]
con.execute("CREATE TABLE s1 AS SELECT * FROM icu WHERE age BETWEEN 18 AND 80")
n1 = con.sql("select count(*) from s1").fetchone()[0]
con.execute("""CREATE TABLE s2 AS SELECT * FROM s1
QUALIFY row_number() OVER (PARTITION BY subject_id ORDER BY intime, stay_id) = 1""")
n2 = con.sql("select count(*) from s2").fetchone()[0]
con.execute("""CREATE TABLE s3 AS SELECT * FROM s2
WHERE outtime >= intime + INTERVAL 12 HOUR
  AND (deathtime IS NULL OR deathtime > intime + INTERVAL 12 HOUR)""")
n3 = con.sql("select count(*) from s3").fetchone()[0]
# unknown death time: in-hospital death flagged but no deathtime
con.execute("""CREATE TABLE cohort AS SELECT *,
   date_diff('second', intime, deathtime)/3600.0 AS t_death_h,
   date_diff('second', intime, dischtime)/3600.0 AS t_disch_h
FROM s3 WHERE NOT (hospital_expire_flag = 1 AND deathtime IS NULL)""")
n4 = con.sql("select count(*) from cohort").fetchone()[0]
log(f"cohort funnel: all={n0} age={n1} first={n2} alive_in_icu_h12={n3} known_death_time={n4}")

# ---------------------------------------------------------------- chartevents
CHART = {
    220045: "hr", 220050: "sbp", 220179: "sbp", 220051: "dbp", 220180: "dbp",
    220210: "rr", 224690: "rr", 220277: "spo2", 223762: "temp", 223761: "temp_f",
    220739: "gcs_e", 223900: "gcs_v", 223901: "gcs_m", 223835: "fio2",
    223849: "vent_mode", 229314: "vent_mode",
}
con.execute("CREATE TABLE chartmap(itemid INT, var VARCHAR)")
con.executemany("INSERT INTO chartmap VALUES (?, ?)", list(CHART.items()))
log("scanning chartevents ...")
con.execute(f"""
CREATE TABLE ce AS
SELECT c.stay_id, m.var, c.charttime::TIMESTAMP charttime, c.valuenum::DOUBLE valuenum, c.value
FROM read_csv('{M}/icu/chartevents.csv.gz', header=true, all_varchar=false,
              types={{'value':'VARCHAR','valueuom':'VARCHAR','warning':'VARCHAR'}}) c
JOIN chartmap m ON m.itemid = c.itemid
JOIN cohort k ON k.stay_id = c.stay_id
WHERE c.charttime::TIMESTAMP >= k.intime AND c.charttime::TIMESTAMP < k.intime + INTERVAL 12 HOUR
""")
log(f"chartevents rows kept: {con.sql('select count(*) from ce').fetchone()[0]}")

# ---------------------------------------------------------------- labevents
LAB = {51006: "bun", 50813: "lactate", 51301: "wbc", 51300: "wbc", 51222: "hb", 50811: "hb",
       50983: "na", 50824: "na", 50971: "k", 50822: "k", 50902: "cl", 50806: "cl",
       50820: "ph", 50802: "be", 50868: "ag"}
con.execute("CREATE TABLE labmap(itemid INT, var VARCHAR)")
con.executemany("INSERT INTO labmap VALUES (?, ?)", list(LAB.items()))
log("scanning labevents ...")
con.execute(f"""
CREATE TABLE le AS
SELECT k.stay_id, m.var, l.charttime::TIMESTAMP charttime, l.valuenum::DOUBLE valuenum
FROM read_csv('{M}/hosp/labevents.csv.gz', header=true,
              types={{'value':'VARCHAR','comments':'VARCHAR','flag':'VARCHAR','priority':'VARCHAR',
                      'valueuom':'VARCHAR','ref_range_lower':'DOUBLE','ref_range_upper':'DOUBLE'}}) l
JOIN labmap m ON m.itemid = l.itemid
JOIN cohort k ON k.hadm_id = l.hadm_id
WHERE l.valuenum IS NOT NULL
  AND l.charttime::TIMESTAMP >= k.intime AND l.charttime::TIMESTAMP < k.intime + INTERVAL 12 HOUR
""")
log(f"labevents rows kept: {con.sql('select count(*) from le').fetchone()[0]}")

# ---------------------------------------------------------------- inputevents (vasopressors)
log("scanning inputevents ...")
con.execute(f"""
CREATE TABLE ie AS
SELECT k.stay_id, e.itemid, e.starttime::TIMESTAMP st, e.endtime::TIMESTAMP et,
       e.rate::DOUBLE rate, e.rateuom, e.patientweight::DOUBLE w, k.intime
FROM read_csv_auto('{M}/icu/inputevents.csv.gz') e
JOIN cohort k ON k.stay_id = e.stay_id
WHERE e.itemid IN (221906, 222315)
  AND e.endtime::TIMESTAMP > k.intime AND e.starttime::TIMESTAMP < k.intime + INTERVAL 12 HOUR
""")
# ---------------------------------------------------------------- procedureevents (invasive vent)
con.execute(f"""
CREATE TABLE pe AS
SELECT k.stay_id, e.starttime::TIMESTAMP st, e.endtime::TIMESTAMP et, k.intime
FROM read_csv_auto('{M}/icu/procedureevents.csv.gz') e
JOIN cohort k ON k.stay_id = e.stay_id
WHERE e.itemid = 225792
  AND e.endtime::TIMESTAMP > k.intime AND e.starttime::TIMESTAMP < k.intime + INTERVAL 12 HOUR
""")
# ---------------------------------------------------------------- outputevents (urine)
UO = (226559, 226560, 226561, 226584, 226563, 226564, 226565, 226567, 226557, 226558, 227489)
con.execute(f"""
CREATE TABLE oe AS
SELECT k.stay_id, e.charttime::TIMESTAMP charttime, e.value::DOUBLE AS value, k.intime
FROM read_csv_auto('{M}/icu/outputevents.csv.gz') e
JOIN cohort k ON k.stay_id = e.stay_id
WHERE e.itemid IN {UO}
  AND e.charttime::TIMESTAMP >= k.intime AND e.charttime::TIMESTAMP < k.intime + INTERVAL 12 HOUR
""")
log("interval tables done")

# ---------------------------------------------------------------- hourly long table
con.execute("CREATE TABLE hours AS SELECT range AS hour FROM range(12)")
con.execute("""
CREATE TABLE long AS
WITH c AS (
  SELECT ce.stay_id, floor(date_diff('second', k.intime, ce.charttime)/3600)::INT AS hour,
         ce.var, ce.charttime, ce.valuenum, ce.value
  FROM ce JOIN cohort k USING (stay_id)
),
simple AS (   -- last value in the hour
  SELECT stay_id, hour,
         CASE var WHEN 'temp_f' THEN 'temp' ELSE var END AS var,
         arg_max(CASE WHEN var = 'temp_f' THEN (valuenum - 32) * 5.0 / 9.0
                      WHEN var = 'fio2' AND valuenum <= 1.0 THEN valuenum * 100
                      ELSE valuenum END, charttime) AS value
  FROM c WHERE var IN ('hr','sbp','dbp','rr','spo2','temp','temp_f','fio2') AND valuenum IS NOT NULL
  GROUP BY ALL
),
gcs AS (      -- total from the last time point with all three components
  SELECT stay_id, hour, 'gcs' AS var, arg_max(tot, charttime) AS value FROM (
    SELECT stay_id, hour, charttime,
           sum(valuenum) FILTER (WHERE var='gcs_e') + sum(valuenum) FILTER (WHERE var='gcs_v')
           + sum(valuenum) FILTER (WHERE var='gcs_m') AS tot
    FROM c WHERE var IN ('gcs_e','gcs_v','gcs_m') GROUP BY stay_id, hour, charttime
  ) WHERE tot IS NOT NULL GROUP BY ALL
),
ventc AS (
  SELECT DISTINCT stay_id, hour, 'vent' AS var, 1.0 AS value FROM c WHERE var = 'vent_mode'
),
ventp AS (
  SELECT DISTINCT pe.stay_id, h.hour, 'vent' AS var, 1.0 AS value
  FROM pe, hours h
  WHERE pe.st < pe.intime + to_hours(h.hour + 1) AND pe.et > pe.intime + to_hours(h.hour)
),
lab AS (
  SELECT le.stay_id, floor(date_diff('second', k.intime, le.charttime)/3600)::INT AS hour,
         le.var, arg_max(le.valuenum, le.charttime) AS value
  FROM le JOIN cohort k USING (stay_id) GROUP BY ALL
),
ne AS (
  SELECT ie.stay_id, h.hour, 'ne' AS var,
         max(CASE WHEN ie.rateuom = 'mcg/kg/min' THEN ie.rate
                  WHEN ie.rateuom = 'mcg/min' AND ie.w > 0 THEN ie.rate / ie.w END) AS value
  FROM ie, hours h
  WHERE ie.itemid = 221906 AND ie.st < ie.intime + to_hours(h.hour + 1) AND ie.et > ie.intime + to_hours(h.hour)
  GROUP BY ALL
),
vaso AS (
  SELECT DISTINCT ie.stay_id, h.hour, 'vaso' AS var, 1.0 AS value
  FROM ie, hours h
  WHERE ie.itemid = 222315 AND ie.st < ie.intime + to_hours(h.hour + 1) AND ie.et > ie.intime + to_hours(h.hour)
),
uo AS (
  SELECT stay_id, floor(date_diff('second', intime, charttime)/3600)::INT AS hour, 'uo' AS var,
         sum(value) AS value
  FROM oe WHERE value IS NOT NULL GROUP BY ALL
)
SELECT * FROM simple UNION ALL SELECT * FROM gcs UNION ALL
SELECT DISTINCT * FROM (SELECT * FROM ventc UNION ALL SELECT * FROM ventp) UNION ALL
SELECT * FROM lab UNION ALL SELECT * FROM ne UNION ALL SELECT * FROM vaso UNION ALL SELECT * FROM uo
""")
con.execute(f"COPY (SELECT * FROM long WHERE hour BETWEEN 0 AND 11 AND value IS NOT NULL) TO '{OUT}/mimic_long.parquet' (FORMAT parquet)")
con.execute(f"""COPY (SELECT subject_id, hadm_id, stay_id, intime, outtime, age, gender, hospital_expire_flag,
                 t_death_h, t_disch_h FROM cohort) TO '{OUT}/mimic_cohort.parquet' (FORMAT parquet)""")
log(con.sql("SELECT var, count(*) n, count(DISTINCT stay_id) stays FROM long GROUP BY var ORDER BY var").fetchall())
log("done")
