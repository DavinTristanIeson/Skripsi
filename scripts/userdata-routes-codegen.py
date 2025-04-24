# Don't touch this script if you're not the developer.

READ_FILE = "routes/userdata/route.template.py"
WRITE_FILE = "routes/userdata/__init__.py"

SPLIT_MARKER = "#CODEGEN SPLIT#"

with open(READ_FILE, "r") as readf:
  contents = readf.read()

header, body = contents.split(SPLIT_MARKER, 2)

TEMPLATE_VALUES = [
  dict(
    label="filter",
    name="filters",
    pascalname="Filters",
    url="filters",
    classname="TableFilter",
    validator="filter_validator",
  ),
  dict(
    label="comparison groups",
    name="comparison_state",
    pascalname="ComparisonState",
    url="comparison-state",
    classname="ComparisonState",
    validator="comparison_state_validator",
  ),
  dict(
    label="dashboard",
    name="dashboard",
    pascalname="Dashboard",
    url="dashboard",
    classname="Dashboard",
    validator="dashboard_validator",
  ),
]

with open(WRITE_FILE, "w") as writef:
  writef.write(header)
  for entry in TEMPLATE_VALUES:
    temp = str(body)
    for key, value in entry.items():
      temp = temp.replace(f"CODEGEN_{key.upper()}", value)
    writef.write(temp)
  