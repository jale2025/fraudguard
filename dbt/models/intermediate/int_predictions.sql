-- {{ config(materialized='table', schema='int') }}

select *
from {{ ref('stg_predictions') }}

UNION All 

select 
    elapsed_sec,
    pc_1,
    pc_2,
    pc_3,
    pc_4,
    pc_5,
    pc_6,
    pc_7,
    pc_8,
    pc_9,
    pc_10,
    pc_11,
    pc_12,
    pc_13,
    pc_14,
    pc_15,
    pc_16,
    pc_17,
    pc_18,
    pc_19,
    pc_20,
    pc_21,
    pc_22,
    pc_23,
    pc_24,
    pc_25,
    pc_26,
    pc_27,
    pc_28,
    amount,
    prediction,
    ingestion_time,
    current_timestamp AT TIME ZONE 'UTC' as dbt_timestamp
from {{ ref('stg_labeled_predictions_queue') }}


{% if target.name == 'dev' %}
    -- limit 1000
{% endif %}