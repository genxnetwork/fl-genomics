iid_split_config = {
    'iid_split_name': 'british_split',
    'n_iid_splits': 5,
}

non_iid_split_config = {
    'non_iid_split_name': 'ethnic_split',
    'non_iid_holdout_ratio': 0.2
}

# Maps ethnic backgrounds to split_ids
# 0 - white british
# 1 - south asian
# 2 - african and carribean
# 3 - chinese
# 4 - others
# 5 - held-out test split from common mixture
split_map = { 
    1001: 0,
    3001: 1,
    3002: 1,
    3003: 1,
    4001: 2,
    4002: 2,
    4003: 2,
    5: 3,
    1: 4,
    1002: 4,
    1003: 4,
    2001: 4,
    2:4,
    2002: 4,
    2003: 4,
    2004: 4,
    3004: 4,
    3: 4,
    4: 4,
    6: 4,    
}


