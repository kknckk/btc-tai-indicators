import os
import re
import glob

def process_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original = content
    # page.tsx specific fixes
    if "page.tsx" in filepath:
        content = content.replace('"{idStr}"', '&quot;{idStr}&quot;')
        content = re.sub(r'h3: \(\{node, \.\.\.props\}\) =>', r'h3: ({node: _node, ...props}) =>', content)
        content = re.sub(r'p: \(\{node, \.\.\.props\}\) =>', r'p: ({node: _node, ...props}) =>', content)
        content = re.sub(r'ol: \(\{node, \.\.\.props\}\) =>', r'ol: ({node: _node, ...props}) =>', content)
        content = re.sub(r'li: \(\{node, \.\.\.props\}\) =>', r'li: ({node: _node, ...props}) =>', content)
        content = re.sub(r'strong: \(\{node, \.\.\.props\}\) =>', r'strong: ({node: _node, ...props}) =>', content)
        content = re.sub(r'code: \(\{node, inline, \.\.\.props\}: any\) =>', r'code: ({node: _node, inline, ...props}: React.ComponentPropsWithoutRef<"code"> & { node?: unknown; inline?: boolean }) =>', content)

    # Replace specific unused icons
    content = re.sub(r',\s*(CheckCircle|Award|AlertCircle|Zap|Layers|ShieldCheck|TrendingUp|ShieldAlert|AreaChart|Area|ZAxis)', '', content)
    content = re.sub(r'(CheckCircle|Award|AlertCircle|Zap|Layers|ShieldCheck|TrendingUp|ShieldAlert|AreaChart|Area|ZAxis),\s*', '', content)
    
    # Paper specific unused variables
    if "Paper8Dashboard.tsx" in filepath:
        content = re.sub(r'const cdd_column = .*?\n', '', content)
        
    # Replace any
    content = content.replace(': any', ': unknown')
    content = content.replace('<any>', '<unknown>')

    if content != original:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Fixed {filepath}")

if __name__ == '__main__':
    files = glob.glob('dashboard/src/**/*.tsx', recursive=True)
    for f in files:
        process_file(f)
