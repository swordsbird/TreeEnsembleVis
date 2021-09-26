import csv

features = ['Account Balance', 'Duration of Credit (month)', 
'Payment Status of Previous Credit', 'Purpose', 'Credit Amount',
'Value Savings/Stocks', 'Length of current employment', 'Installment per cent',
'Sex & Marital Status', 'Guarantors', 'Duration in Current address',
'Most valuable available asset', 'Age (years)', 'Concurrent Credits',
'Type of apartment', 'No of Credits at this Bank', 'Occupation',
'No of dependents', 'Telephone', 'Foreign Worker', 'Creditability']

reassign = {
    'Creditability' : ['No', 'Yes' ], 
    'Payment Status of Previous Credit' : [
        'no credits taken/ all credits paid back duly',
        'all credits at this bank paid back duly',
        'existing credits paid back duly till now',
        'delay in paying off in the past',
        'critical account/ other credits existing (not at this bank)',
    ],
    'Purpose' : [
        'car (new)',
        'car (used)',
        'furniture/equipment',
        'radio/television',
        'domestic appliances',
        'repairs',
        'education',
        '(vacation - does not exist?)',
        'retraining',
        'business',
        'others'
    ],
    'Value Savings/Stocks': [
        '< 100 DM',
        '100 <= ... < 500 DM',
        '500 <= ... < 1000 DM',
        '>= 1000 DM',
        'unknown/ no savings account',
    ],
    'Length of current employment': [
        'unemployed',
        '< 1 year',
        '1 <= ... < 4 years',
        '4 <= ... < 7 years',
        '>= 7 years'
    ],
    'Sex & Marital Status': [
        'male : divorced/separated',
        'female : divorced/separated/married',
        'male : single',
        'male : married/widowed',
        'female : single'
    ],
    'Guarantors': [
        'none',
        'co-applicant',
        'guarantor'
    ],
    'Most valuable available asset': [
        'real estate',
        'building society savings agreement/ life insurance',
        'car or other',
        'unknown / no property'
    ],
    'Concurrent Credits': ['bank', 'stores', 'none'],
    'Type of apartment': ['rent', 'own', 'for free'],
    'Occupation': [
        'unemployed/ unskilled - non-resident',
        'unskilled - resident',
        'skilled employee / official',
        'management/ self-employed/ highly qualified employee/ officer'
    ],
    'Telephone': ['No', 'Yes'],
    'Foreign Worker': ['Yes', 'No']
}
