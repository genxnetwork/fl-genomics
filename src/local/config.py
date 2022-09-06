node_size_dict = {
    'uneven_split': {
        0: 6103,
        1: 1199,
        2: 171578,
        3: 85813,
        4: 42884,
        5: 21444,
        6: 10717,
        7: 3432
    },
    'ethnic_split': {
        0: 343858,
        1: 6139,
        2: 6086,
        3: 1201,
        4: 30317 
    },
    'region_split': {
        0: 47971,
        1: 70311,
        2: 57158,
        3: 30528,
        4: 38283,
        5: 10854,
        6: 41348,
        7: 29777,
        8: 29253,
        9: 19399,
        10: 36027
    },
    'tg_split': {**{
        0: 370000,
        1: 7900,
        2: 7100,
        3: 2056,
        4: 400,
        }, **{i: 7500 for i in range(5, 15)}
                }
}

node_name_dict = {
    'uneven_split': {i: 'WB' for i in range(8)},
    'ethnic_split': {
        0: 'WB',
        1: 'SA',
        2: 'AC',
        3: 'CN',
        4: 'mix'
    },
    'region_split': {
        0: 'North East (England)',
        1: 'North West (England)',
        2: 'Yorkshire and The Humber',
        3: 'East Midlands (England)',
        4: 'West Midlands (England)',
        5: 'East of England',
        6: 'London',
        7: 'South East (England)',
        8: 'South West (England)',
        9: 'Wales',
        10: 'Scotland'
    },
    'tg_split': {**{
        0: "EUR",
        1: "SAS",
        2: "AFR",
        3: "EAS",
        4: "AM"
        }, **{i: f'EUR_{i}' for i in range(5, 15)}}
}
