#!/usr/bin/env bash
# Read-only metadata search (schemas, tables, column names only — no row data) on the local PostgreSQL.
# Looks for anything that could hold ICU admission / transfer timestamps.
set -uo pipefail
pg_ro() {
  if [ -n "${SUDO_PW:-}" ]; then
    sudo -S -p "" -u postgres env PGOPTIONS="-c default_transaction_read_only=on" "$@" <<<"$SUDO_PW"
  else
    sudo -u postgres env PGOPTIONS="-c default_transaction_read_only=on" "$@"
  fi
}
echo "== databases"
pg_ro psql -tAc "SELECT datname FROM pg_database WHERE NOT datistemplate ORDER BY 1"
for db in $(pg_ro psql -tAc "SELECT datname FROM pg_database WHERE NOT datistemplate AND datname NOT IN ('postgres') ORDER BY 1"); do
  echo "== $db: schemas / tables (estimated rows)"
  pg_ro psql -d "$db" -tAc "
    SELECT n.nspname || '.' || c.relname || ' (' || c.reltuples::bigint || ')'
    FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
    WHERE c.relkind IN ('r','v','m','p') AND n.nspname NOT IN ('pg_catalog','information_schema') AND n.nspname NOT LIKE 'pg_toast%'
    ORDER BY 1"
  echo "== $db: columns that may carry ICU / transfer / unit timing"
  pg_ro psql -d "$db" -tAc "
    SELECT table_schema || '.' || table_name || '.' || column_name || ' : ' || data_type
    FROM information_schema.columns
    WHERE table_schema NOT IN ('pg_catalog','information_schema')
      AND (column_name ILIKE '%icu%' OR column_name ILIKE '%transfer%' OR column_name ILIKE '%ward%' OR column_name ILIKE '%unit%'
           OR column_name ILIKE '%intime%' OR column_name ILIKE '%outtime%' OR column_name ILIKE '%in_time%' OR column_name ILIKE '%out_time%'
           OR column_name ILIKE '%move%' OR column_name ILIKE '%location%' OR column_name ILIKE '%dept%' OR column_name ILIKE '%room%'
           OR column_name ILIKE '%bed%' OR column_name ILIKE '%stay%')
    ORDER BY 1"
done
