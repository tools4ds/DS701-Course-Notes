import sys
import json

def remove_div_fences(cell):
    if cell['cell_type'] == 'markdown':
        cell['source'] = ''.join(line for line in cell['source'] if not line.startswith(':::'))
    return cell

def main():
    notebook = json.load(sys.stdin)
    notebook['cells'] = [remove_div_fences(cell) for cell in notebook['cells']]
    json.dump(notebook, sys.stdout)

if __name__ == '__main__':
    main()
