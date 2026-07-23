#!/usr/bin/env python
# Fix script to remove RIPENESS_SHELF_LIFE references

with open('streamlit_app_new.py', 'r') as f:
    content = f.read()

# Remove RIPENESS_SHELF_LIFE dictionary definition
content = content.replace("""RIPENESS_SHELF_LIFE = {
    "breaker": "5-7 days (Break-stage)",
    "defect": "0 days ❌ REJECT",
    "green": "7-10 days (Unripe)",
    "red": "1-3 days (Fully Ripe)",
    "turning": "3-5 days (Transitioning)"
}

""", "")

# Remove shelf_life variable assignments
lines = content.split('\n')
new_lines = []
for line in lines:
    if 'shelf_life = RIPENESS_SHELF_LIFE' in line:
        continue
    if 'Shelf Life:' in line and 'st.write' in line:
        continue
    if '"Shelf Life": RIPENESS_SHELF_LIFE' in line:
        continue
    new_lines.append(line)

content = '\n'.join(new_lines)

with open('streamlit_app_new.py', 'w') as f:
    f.write(content)

print('✅ Fixed - Removed all RIPENESS_SHELF_LIFE references')
