-- Intermediate layer: Date + Time -> Datetime -> Year/Month/Day/Hour
-- and business-rule filtering (valid ranges).

with stg as (

    select * from {{ ref('stg_earthquakes') }}

),

parsed as (

    select
        id,
        latitude,
        longitude,
        type,
        depth,
        magnitude,
        try_strptime(date_str || ' ' || time_str, '%m/%d/%Y %H:%M:%S') as event_datetime
    from stg

),

filtered as (

    select
        id,
        latitude,
        longitude,
        type,
        depth,
        magnitude,
        event_datetime,
        extract(year  from event_datetime) as year,
        extract(month from event_datetime) as month,
        extract(day   from event_datetime) as day,
        extract(hour  from event_datetime) as hour
    from parsed
    where event_datetime is not null
      and depth between 0 and 800
      and magnitude between 0 and 10

)

select * from filtered
