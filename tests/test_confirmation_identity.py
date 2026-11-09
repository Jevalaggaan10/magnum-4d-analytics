import json
import unittest

from forecast_evaluation import confirmation_status, promotion_decision
from forecast_identity import procedure_hash, record_identity
from forecast_fixtures import ROOT, settings


def record(date='20261107'):
    return {'for_date': date, 'confirmation_version': 'series-a',
            'procedure_sha256': 'a' * 64, 'execution_revision_sha256': 'b' * 64,
            'coverage': 1, 'status': 'verified'}


class ConfirmationIdentityTests(unittest.TestCase):
    def test_same_label_with_changed_procedure_is_rejected(self):
        rows = [record('20261107'), dict(record('20261108'), procedure_sha256='c' * 64)]
        self.assertEqual(confirmation_status(rows)['status'], 'invalid-identity')

    def test_identity_cannot_be_omitted_even_before_checkpoint(self):
        self.assertEqual(confirmation_status([{'for_date': '20261107', 'coverage': 1}])['status'],
                         'invalid-identity')

    def test_duplicate_target_does_not_count_twice(self):
        self.assertEqual(confirmation_status([record(), record()])['status'], 'invalid-identity')

    def test_matching_series_remains_pending_without_comparisons(self):
        rows = [record('20261107'), record('20261108')]
        result = promotion_decision(rows, development_qualified=True, correctness_verified=True)
        self.assertEqual(result['status'], 'pending')
        self.assertEqual(result['scheduled_targets'], 2)
        self.assertNotIn('intervals', result)

    def test_failure_placeholder_counts_without_invented_artifact(self):
        rows = [dict(record(), coverage=0, status='missing-forecast')]
        self.assertEqual(confirmation_status(rows, checkpoint=1)['status'], 'incomplete-coverage')

    def test_operational_flags_do_not_change_scientific_identity(self):
        original = settings()
        revised = dict(original, correctness_verified=True, requested_date='20261108',
                       correctness_limit='hosted checks completed')
        self.assertEqual(procedure_hash(original), procedure_hash(revised))
        self.assertNotEqual(procedure_hash(original),
                            procedure_hash(dict(original, shadow={'model': 'digit'})))

    def test_explicit_series_keeps_archive_separate_and_missing_schedule_in_denominator(self):
        config = settings()
        active = {**record('20261108'), **record_identity(config)}
        active['forecasts']={'champion':{'metadata':{'configuration':{'model':'uniform'},
                                'training_operator':'damacai','feature_version':'lagged-digits-v1'}}}
        archive = dict(record('20261107'), confirmation_version='archived')
        status = confirmation_status([archive, active], checkpoint=2, settings=config)
        self.assertEqual(status['status'], 'incomplete-coverage')
        self.assertEqual(status['scheduled_targets'], 2)

    def test_unapproved_revision_is_not_accepted_as_same_procedure(self):
        config = settings()
        row = {**record(), **record_identity(config), 'execution_revision_sha256': 'f' * 64}
        self.assertEqual(confirmation_status([row], settings=config)['status'], 'invalid-identity')

    def test_reviewed_legacy_record_is_accepted_but_changed_picks_are_rejected(self):
        config=json.loads((ROOT/'forecast_config.json').read_text())
        state=json.loads((ROOT/'predictions.json').read_text())
        row=state['probability_forecasts']['20261107']
        self.assertEqual(confirmation_status([row],settings=config)['status'],'pending')
        row['forecasts']['champion']['picks'][0]='9876'
        self.assertEqual(confirmation_status([row],settings=config)['status'],'invalid-identity')
