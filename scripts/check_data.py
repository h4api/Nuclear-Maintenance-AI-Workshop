"""Dependency-free dataset validation; no model execution."""
from collections import Counter
from common import ROOT, read_rows, schema_ok, QUEUES

def main():
    seen_ids, seen_inputs = set(), set()
    references = {'training_record': 'course_id', 'vendor_document': 'equipment_identifier',
                  'maintenance_record': 'work_order_id', 'document_change_request': 'document_id'}
    labels = {'course_id': 'Course ID', 'equipment_identifier': 'Equipment identifier',
              'work_order_id': 'Work order ID', 'document_id': 'Document ID'}
    for split, expected in [('train', 60), ('dev', 20), ('test', 20)]:
        rows = read_rows(ROOT / 'data' / (split + '.jsonl'))
        assert len(rows) == expected, (split, len(rows))
        for row in rows:
            assert row['id'] not in seen_ids and row['input'] not in seen_inputs, 'Split overlap'
            seen_ids.add(row['id']); seen_inputs.add(row['input'])
            target = row['target']
            assert schema_ok(target), row['id']
            assert target['review_queue'] == QUEUES[target['category']], row['id']
            missing = []
            if 'Department: not supplied.' in row['input']:
                assert target['department'] is None
                missing.append('department')
            else:
                assert 'Department: ' + target['department'] + '.' in row['input']
            reference = references.get(target['category'])
            if reference and labels[reference] + ': not supplied.' in row['input']:
                missing.append(reference)
            assert target['missing_information'] == sorted(missing), row['id']
        counts = Counter(r['target']['category'] for r in rows)
        assert len(counts) == 5 and len(set(counts.values())) == 1
        print(split, len(rows), 'records:', dict(counts))
    print('PASS: counts, schemas, queue mappings, missing-field labels and disjoint inputs/IDs.')
    print('Description/category semantics still require instructor review; templates are correlated.')

if __name__ == '__main__':
    main()
