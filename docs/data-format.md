# Bring your own museum data

MuseKG reads a top-level JSON array of object records. Each record requires a stable
`opacObjectId`. Descriptive fields are supplied through `opacObjectFieldSets`; relations
and image labels are optional.

```json
{
  "opacObjectId": "LOCAL-001",
  "opacObjectFieldSets": [
    {"identifier": "name", "opacObjectFields": [{"value": "Object title"}]},
    {"identifier": "material_desc", "opacObjectFields": [{"value": "Material"}]}
  ],
  "relationshipsCollection": {
    "relationships": [
      {
        "relationshipId": "object_prod_pri_org",
        "relatedRecords": [
          {"relatedRecordId": "ORG-1", "relatedRecordType": "organisation", "title": "Producer"}
        ]
      }
    ]
  },
  "imagesCollection": {"images": []}
}
```

Supported object field identifiers are `name`, `description`, `material_desc`,
`accession_no`, `credit_line`, `production_date`, `measurements`, `object_type`,
`collection`, `history_cat`, and `museum`.

Optional entity input has the shape `{"results": [{"object_id": "...", "entities":
{"TYPE": [{"name": "..."}]}}]}`. Optional OCR input has the shape `{"results":
[{"image_name": "OBJECT_ID.jpg", "extracted_text": "..."}]}`.

MuseKG reads these files locally. The graph builder does not upload them. If you enable
an external model or API in downstream code, review that provider's data-handling terms
before sending sensitive museum content.
