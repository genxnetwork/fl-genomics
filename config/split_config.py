uniform_split_config = {
    'uniform_split_name': 'uniform_split',
    'n_nodes': 5,
}

non_iid_split_name = 'ethnic_split'
uneven_split_shares_list = [0.01777, 0.00349, 1/2, 1/4, 1/8, 1/16 , 1/32]

heterogeneous_split_name = 'heterogeneous_split'
n_heterogeneous_nodes = 4

# Maps ethnic backgrounds to split_ids
# 0 - white british
# 1 - south asian
# 2 - african and carribean
# 3 - chinese
# 4 - others
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

ethnic_background_name_map = {
    1:	'White',
    1001:	'British',
    2001:	'White and Black Caribbean',
    3001:	'Indian',
    4001:	'Caribbean',
    2:	'Mixed',
    1002:	'Irish',
    2002:	'White and Black African',
    3002:	'Pakistani',
    4002:	'African',
    3:	'Asian or Asian British',
    1003:	'Any other white background',
    2003:	'White and Asian',
    3003:	'Bangladeshi',
    4003:	'Any other Black background',
    4:	'Black or Black British',
    2004:	'Any other mixed background',
    3004:	'Any other Asian background',
    5: 'Chinese',
    6:	'Other ethnic group'
}

heterogeneous_split_codes = [1, 1001, 1002, 1003]

random_seed = 32
