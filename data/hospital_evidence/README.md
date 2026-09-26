# Hospital evidence and provenance

This directory contains record-level supporting evidence used to construct and validate the tertiary Grade A hospital database for the study.

The files are provided primarily for transparency and auditability of hospital inclusion, accreditation timing, campus configuration, and bed-capacity reconstruction.

## `hospital_legal_entity_evidence.csv`

Record-level evidence for the hospital legal-entity database.

| Column | Description |
|---|---|
| `legal_entity_id` | Unique identifier for each hospital legal entity. |
| `hospital_legal_entity_name` | Official or commonly used Chinese name of the hospital legal entity. |
| `aliases_or_campus_names` | Alternative hospital names, historical names, or associated campus names used for record matching. |
| `institution_category` | Institutional category of the hospital. |
| `province` | Province-level administrative unit in Chinese. |
| `province_en` | Province-level administrative unit in English. |
| `city` | Prefecture-level administrative unit in Chinese. |
| `city_en` | Prefecture-level administrative unit in English. |
| `tertiary_grade_a_analysis_start_year` | First year from which the hospital is treated as tertiary Grade A in the study database. A documented accreditation year was used when available. For hospitals already confirmed as tertiary Grade A before 2014 but without a traceable exact accreditation year, the value was set to 2014, the first year of the study period. This field therefore denotes the analysis start year of Grade A status, not necessarily the original historical accreditation year. |
| `primary_source_urls` | Primary traceable source URL(s) supporting hospital identity, eligibility, accreditation status, or accreditation timing. |

## `hospital_campus_capacity_evidence.csv`

Record-level evidence used to reconstruct hospital campuses and bed capacity over time.

| Column | Description |
|---|---|
| `campus_id` | Unique identifier for each hospital campus or spatial service location. |
| `province` | Province-level administrative unit in Chinese. |
| `province_en` | Province-level administrative unit in English. |
| `city` | Prefecture-level administrative unit in Chinese. |
| `city_en` | Prefecture-level administrative unit in English. |
| `hospital_name` | Name of the corresponding hospital legal entity. |
| `campus_name` | Name of the hospital campus. |
| `evidence_year` | Year to which the supporting evidence refers. |
| `bed_value` | Bed-capacity value reported or supported by the source. |
| `status_event` | Documented operational or capacity-related event associated with the record, where applicable. |
| `source_urls` | Traceable source URL(s) supporting the campus, operational status, or bed-capacity record. |

## Notes

Hospital and campus names are retained in Chinese to facilitate matching with the original institutional and government sources. Province and prefecture-level administrative units are provided in both Chinese and English.

These provenance tables are intended to document the evidentiary basis of the hospital database. They are not direct inputs to the default public reproduction workflow.