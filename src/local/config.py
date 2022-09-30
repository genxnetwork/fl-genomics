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
                },
    'ac_split': {
        0: 9654,
        1: 17917,
        2: 33655,
        3: 16446,
        4: 14077,
        5: 10324,
        6: 20747,
        7: 12444,
        8: 14090,
        9: 21598,
        10: 29736,
        11: 21083,
        12: 15053,
        13: 25828,
        14: 24100,
        15: 10868,
        16: 22689,
        17: 19482,
        18: 13173,
        19: 352946
 
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
        }, **{i: f'EUR_{i}' for i in range(5, 15)}},
    'ac_split': {
        0: 'Barts',
        1: 'Birmingham',
        2: 'Bristol',
        3: 'Bury',
        4: 'Cardiff',
        5: 'Cheadle (revisit)',
        6: 'Croydon',
        7: 'Edinburgh',
        8: 'Glasgow',
        9: 'Hounslow',
        10: 'Leeds',
        11: 'Liverpool',
        12: 'Middlesborough',
        13: 'Newcastle',
        14: 'Nottingham',
        15: 'Oxford',
        16: 'Reading',
        17: 'Sheffield',
        18: 'Stoke',
        19: 'Centralized'
     }
}
