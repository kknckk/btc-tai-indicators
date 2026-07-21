import re
import os

def fix_page():
    f = 'dashboard/src/app/literatura/[paperId]/page.tsx'
    with open(f, 'r') as fp: c = fp.read()
    c = c.replace('"{idStr}"', '&quot;{idStr}&quot;')
    c = re.sub(r'h3: \(\{node, \.\.\.props\}\)', r'h3: ({...props}: any)', c)
    c = re.sub(r'p: \(\{node, \.\.\.props\}\)', r'p: ({...props}: any)', c)
    c = re.sub(r'ol: \(\{node, \.\.\.props\}\)', r'ol: ({...props}: any)', c)
    c = re.sub(r'li: \(\{node, \.\.\.props\}\)', r'li: ({...props}: any)', c)
    c = re.sub(r'strong: \(\{node, \.\.\.props\}\)', r'strong: ({...props}: any)', c)
    c = re.sub(r'code: \(\{node, inline, \.\.\.props\}: any\)', r'code: ({inline, ...props}: any)', c)
    with open(f, 'w') as fp: fp.write(c)

def fix_any(files):
    for f in files:
        if not os.path.exists(f): continue
        with open(f, 'r') as fp: c = fp.read()
        c = c.replace(': any', ': unknown')
        c = c.replace('<any>', '<unknown>')
        c = c.replace('props: unknown', 'props: any') # don't break react props too much if they were any
        with open(f, 'w') as fp: fp.write(c)

def fix_unused():
    # Paper10
    f = 'dashboard/src/components/literature/Paper10Dashboard.tsx'
    with open(f, 'r') as fp: c = fp.read()
    c = c.replace('CheckCircle, Award,', '')
    with open(f, 'w') as fp: fp.write(c)
    
    # Paper2
    f = 'dashboard/src/components/literature/Paper2Dashboard.tsx'
    with open(f, 'r') as fp: c = fp.read()
    c = c.replace(' ZAxis,', '')
    with open(f, 'w') as fp: fp.write(c)
    
    # Paper3
    f = 'dashboard/src/components/literature/Paper3Dashboard.tsx'
    with open(f, 'r') as fp: c = fp.read()
    c = c.replace(' AlertCircle,', '')
    with open(f, 'w') as fp: fp.write(c)
    
    # Paper5
    f = 'dashboard/src/components/literature/Paper5Dashboard.tsx'
    with open(f, 'r') as fp: c = fp.read()
    c = c.replace('CheckCircle, AlertCircle, ', '')
    with open(f, 'w') as fp: fp.write(c)
    
    # Paper6
    f = 'dashboard/src/components/literature/Paper6Dashboard.tsx'
    with open(f, 'r') as fp: c = fp.read()
    c = c.replace('Zap, Layers, ', '')
    with open(f, 'w') as fp: fp.write(c)
    
    # Paper7
    f = 'dashboard/src/components/literature/Paper7Dashboard.tsx'
    with open(f, 'r') as fp: c = fp.read()
    c = c.replace('CheckCircle, ShieldCheck, Layers,', '')
    with open(f, 'w') as fp: fp.write(c)
    
    # Paper8
    f = 'dashboard/src/components/literature/Paper8Dashboard.tsx'
    with open(f, 'r') as fp: c = fp.read()
    c = re.sub(r'const cdd_column = .*?\n', '', c)
    with open(f, 'w') as fp: fp.write(c)
    
    # Paper9
    f = 'dashboard/src/components/literature/Paper9Dashboard.tsx'
    with open(f, 'r') as fp: c = fp.read()
    c = c.replace('AreaChart, Area,', '')
    c = c.replace('Layers, TrendingUp, ShieldAlert,', '')
    with open(f, 'w') as fp: fp.write(c)
    
    # Paper1View
    f = 'dashboard/src/components/literature/Paper1View.tsx'
    if os.path.exists(f):
        with open(f, 'r') as fp: c = fp.read()
        c = c.replace('"{details.title}"', '&quot;{details.title}&quot;')
        c = c.replace('"{details.summary}"', '&quot;{details.summary}&quot;')
        c = c.replace('let i = 0;', '')
        with open(f, 'w') as fp: fp.write(c)
        
    # Chart
    f = 'dashboard/src/components/Chart.tsx'
    if os.path.exists(f):
        with open(f, 'r') as fp: c = fp.read()
        c = c.replace('data: any[];', 'data: unknown[];')
        c = c.replace('series: any[];', 'series: unknown[];')
        with open(f, 'w') as fp: fp.write(c)

    # ApiTester
    f = 'dashboard/src/components/ApiTester.tsx'
    if os.path.exists(f):
        with open(f, 'r') as fp: c = fp.read()
        c = c.replace('{/* Test API */}', '')
        c = c.replace(': any', ': unknown')
        with open(f, 'w') as fp: fp.write(c)

if __name__ == "__main__":
    fix_page()
    fix_unused()
    fix_any([
        'dashboard/src/components/literature/Paper1Dashboard.tsx',
        'dashboard/src/components/literature/Paper2Dashboard.tsx',
        'dashboard/src/components/literature/Paper3Dashboard.tsx',
        'dashboard/src/components/literature/Paper4Dashboard.tsx',
        'dashboard/src/components/literature/Paper5Dashboard.tsx',
        'dashboard/src/components/literature/Paper6Dashboard.tsx',
        'dashboard/src/components/literature/Paper7Dashboard.tsx',
        'dashboard/src/components/literature/Paper8Dashboard.tsx',
        'dashboard/src/components/literature/Paper9Dashboard.tsx',
        'dashboard/src/components/literature/Paper10Dashboard.tsx',
        'dashboard/src/components/literature/Paper1View.tsx'
    ])
