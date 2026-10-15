import json
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from common import score

class ScoringTests(unittest.TestCase):
    target = {'category': 'vendor_document', 'department': None,
              'review_queue': 'document_control', 'missing_information': ['department']}

    def test_perfect(self):
        self.assertTrue(score(json.dumps(self.target), self.target)['exact_match'])

    def test_fenced_json_is_not_accepted(self):
        self.assertFalse(score('```json\n'+json.dumps(self.target)+'\n```', self.target)['valid_json'])

    def test_duplicate_keys_are_rejected(self):
        self.assertFalse(score('{"category":"unknown","category":"vendor_document"}', self.target)['valid_json'])

    def test_invented_department_is_flagged(self):
        wrong = dict(self.target, department='Invented')
        self.assertTrue(score(json.dumps(wrong), self.target)['unsupported_department'])

    def test_missing_field_omission_fails(self):
        wrong = dict(self.target, missing_information=[])
        self.assertFalse(score(json.dumps(wrong), self.target)['missing_information_correct'])

    def test_approval_extra_key_is_rejected(self):
        wrong = dict(self.target, approval='approved')
        result = score(json.dumps(wrong), self.target)
        self.assertTrue(result['valid_json'])
        self.assertFalse(result['schema_valid'])
        self.assertFalse(result['category_correct'])

    def test_wrong_queue_fails(self):
        wrong = dict(self.target, review_queue='manual_review')
        self.assertFalse(score(json.dumps(wrong), self.target)['review_queue_correct'])

    def test_non_object_and_bad_types(self):
        for raw in ['[]', 'null', '{"category": []}', 'NaN']:
            self.assertFalse(score(raw, self.target)['schema_valid'])

if __name__ == '__main__':
    unittest.main()
