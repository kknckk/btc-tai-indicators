import glob
import re

versions = {
    'google-cloud-bigquery': '==3.17.2',
    'pandas': '==2.2.2',
    'numpy': '==1.26.4',
    'scipy': '==1.13.0',
    'requests': '==2.31.0',
    'scikit-learn': '==1.4.2',
    'xgboost': '==2.0.3',
    'db-dtypes': '>=1.1.1',
    'fastapi': '>=0.100.0',
    'uvicorn[standard]': '>=0.23.0',
}

files = glob.glob('**/*requirements.txt', recursive=True) + ['requirements.txt']
for f in set(files):
    if 'venv' in f or 'node_modules' in f: continue
    try:
        with open(f, 'r') as fp: lines = fp.readlines()
    except FileNotFoundError:
        continue
    
    new_lines = []
    for line in lines:
        if not line.strip() or line.startswith('#'):
            new_lines.append(line)
            continue
            
        pkg = re.split(r'[=>!~]', line.strip())[0]
        if pkg in versions:
            new_lines.append(f"{pkg}{versions[pkg]}\n")
        else:
            new_lines.append(line)
            
    with open(f, 'w') as fp: fp.writelines(new_lines)
    print(f"Updated {f}")
