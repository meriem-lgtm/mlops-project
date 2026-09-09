-- Staging layer: light cleaning.
-- Source: the raw_earthquakes table loaded by dlt.

with source as (

    select * from {{ source('raw', 'raw_earthquakes') }}

),

renamed as (

    select
        id,
        date        as date_str,
        time        as time_str,
        latitude,
        longitude,
        type,
        depth,
        magnitude
    from source
    where latitude is not null
      and longitude is not null
      and magnitude is not null
      and depth is not null
      and latitude between -90 and 90
      and longitude between -180 and 180
      and depth between 0 and 800
      and magnitude between 0 and 10

)

select * from renamed