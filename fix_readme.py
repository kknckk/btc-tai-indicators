with open("README.md", "r") as f:
    c = f.read()

c = c.replace('## Project Context for Future AI Agents ("Vibe Coding")', '## Project Context')
c = c.replace('**Dear AI Agent:** This repository contains', 'This repository contains')
c = c.replace('### What Needs To Be Done Next (Action Plan for You)', '### What Needs To Be Done Next')
with open("README.md", "w") as f:
    f.write(c)

with open("ROADMAP.md", "r") as f:
    c = f.read()

c = c.replace('Witaj Agencie! Ten dokument opisuje plan działania', 'Ten dokument opisuje plan działania')
with open("ROADMAP.md", "w") as f:
    f.write(c)
