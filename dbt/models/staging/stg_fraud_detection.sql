{{ config(materialized='table', schema='stg') }}

select 
    "Time" as elapsed_sec,
    "V1" as pc_1,
    "V2" as pc_2,
    "V3" as pc_3,
    "V4" as pc_4,
    "V5" as pc_5,
    "V6" as pc_6,
    "V7" as pc_7,
    "V8" as pc_8,
    "V9" as pc_9,
    "V10" as pc_10,
    "V11" as pc_11,
    "V12" as pc_12,
    "V13" as pc_13,
    "V14" as pc_14,
    "V15" as pc_15,
    "V16" as pc_16,
    "V17" as pc_17,
    "V18" as pc_18,
    "V19" as pc_19,
    "V20" as pc_20,
    "V21" as pc_21,
    "V22" as pc_22,
    "V23" as pc_23,
    "V24" as pc_24,
    "V25" as pc_25,
    "V26" as pc_26,
    "V27" as pc_27,
    "V28" as pc_28,
    "Amount" as amount,
    "Class" as class,
    ingestion_time,
    current_timestamp AT TIME ZONE 'UTC' as dbt_timestamp
from {{ source('src_fraud_detection', 'transactions') }}

{% if target.name == 'dev' %}
    limit 1000
{% endif %}