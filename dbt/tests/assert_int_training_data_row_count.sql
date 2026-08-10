-- The training set is a UNION ALL of the two staging models, so its row count has to
-- be exactly their sum. Returning any row fails the test.
--
-- This guards two mistakes that are otherwise silent: a UNION ALL turned into a UNION,
-- which would deduplicate legitimately identical transactions, and a column added to
-- one branch of the union but not the other.

with expected as (

    select
        (select count(*) from {{ ref('stg_fraud_detection') }})
        + (select count(*) from {{ ref('stg_labeled_predictions_queue') }}) as row_count

),

actual as (

    select count(*) as row_count from {{ ref('int_training_data') }}

)

select
    expected.row_count as expected_row_count,
    actual.row_count as actual_row_count
from expected
cross join actual
where expected.row_count != actual.row_count
