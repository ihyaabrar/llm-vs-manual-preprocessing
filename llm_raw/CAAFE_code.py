# (Comorbidity and complication interaction: patients with both no comorbidity and no complication may have simpler disease management)
# Usefulness: Combined absence of comorbidities and complications indicates simpler health profile, likely higher adherence.
# Input samples: 'COMORBIDITY_NO_COMORBIDITY': [1.0, 1.0, 1.0], 'COMPLICATIONDEVELOPMENT_NO_COMPLICATION': [1.0, 1.0, 1.0]
df['SIMPLE_CASE'] = (df['COMORBIDITY_NO_COMORBIDITY'] == 1) & (df['COMPLICATIONDEVELOPMENT_NO_COMPLICATION'] == 1)
df['SIMPLE_CASE'] = df['SIMPLE_CASE'].astype(int)