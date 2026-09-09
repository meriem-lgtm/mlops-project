-- Marts layer: final ML-ready table.
-- Mirrors the Python feature logic in src/features/engineering.py so
-- the two stay conceptually aligned (keep them in sync if you change one).

with base as (

    select * from {{ ref('int_earthquakes_clean') }}

),

featured as (

    select
        id,
        latitude,
        longitude,
        depth,
        magnitude,
        type,
        year,
        month,
        day,
        hour,

        -- region_enc: coarse geographic bucket (see region_enc() in Python)
        case
            when latitude >= 0 and longitude between -30 and 60 then 0
            when latitude >= 0 and longitude > 60               then 1
            when latitude >= 0 and longitude < -30              then 2
            when latitude < 0 and longitude between -30 and 60  then 3
            when latitude < 0 and longitude > 60                then 4
            else 5
        end as region_enc,

       floor((latitude + 90) / 10) as lat_bin,
floor((longitude + 180) / 10) as lon_bin,
        sqrt(power(latitude, 2) + power(longitude, 2)) as distance_center,
        case when depth > 300 then 1 else 0 end          as is_deep,

        case
    when magnitude < 5 then 'Faible'
    when magnitude < 6.5 then 'Moyen'
    else 'Fort'
end as magnitude_class

    from base

)

select * from featured
