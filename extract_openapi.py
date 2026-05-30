import json
from smartwave_ai.app import app

# Generate the OpenAPI schema
openapi_schema = app.openapi()

# Save it to a file
with open("openapi.json", "w", encoding="utf-8") as f:
    json.dump(openapi_schema, f, indent=2, ensure_ascii=False)

print("OpenAPI schema successfully extracted to openapi.json!")
print("Exposed paths:")
for path in openapi_schema.get("paths", {}).keys():
    print(f" - {path}")
