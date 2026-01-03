import pandas as pd


df_positive = pd.read_csv('./proteases/cleave_pattern_one_letter_aa_S01.247.csv')
df_negative = pd.read_csv('./proteases/negative_pattern_one_letter_aa_S01.247.csv')

positive_cleave_patterns = df_positive['cleave_pattern']
positive_merops_id = df_positive['merops_id']
positive_uniprot_id = df_positive['uniprot_id']

negative_cleave_patterns = df_negative['negative_pattern']
negative_merops_id = df_negative['merops_id']
negative_uniprot_id = df_negative['uniprot_id']

for i in range(len(positive_cleave_patterns)):
    with open('566_S01.247_training_postive.fasta', 'a', encoding='utf-8') as f:
        f.write('> {}, {} \n {} \n'.format(positive_merops_id[i], positive_uniprot_id[i], positive_cleave_patterns[i]))
for i in range(len(negative_cleave_patterns)):
    with open('566_S01.247_training_negative.fasta', 'a', encoding='utf-8') as f:
        f.write('> {}, {} \n {} \n'.format(negative_merops_id[i], negative_uniprot_id[i], negative_cleave_patterns[i]))

print("END.")