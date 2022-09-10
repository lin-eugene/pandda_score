from fileinput import filename
import gemmi
import sys

filename_map = sys.argv[1]
filetype = filename_map.split('.')[-1]

if not filetype == 'ccp4':
    exit

map = gemmi.read_ccp4_map(filename_map,setup=True)

print(map.grid)
print(map.grid.spacegroup)
print(map.grid.unit_cell)