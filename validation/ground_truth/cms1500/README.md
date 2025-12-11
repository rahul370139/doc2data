# CMS-1500 Ground Truth

Store JSON ground truth files matching the test PDFs in `data/sample_docs/cms1500_test/`.

Expected format:
```json
{
  "patient_name": "JOHN DOE",
  "patient_dob": "01/15/1985",
  "insurance_id": "ABC12345",
  "billing_npi": "1234567893",
  "...": "..."
}
```

Use the same basename as the PDF, e.g. `cms1500_handwritten_01.json` for `cms1500_handwritten_01.pdf`.
