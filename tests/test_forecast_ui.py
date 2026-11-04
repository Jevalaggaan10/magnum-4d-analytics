import importlib.util
import unittest
from pathlib import Path


@unittest.skipUnless(importlib.util.find_spec('streamlit'),'UI environment only')
class ForecastUITests(unittest.TestCase):
    def test_home_shows_categorical_probability_and_explicit_target(self):
        from streamlit.testing.v1 import AppTest
        at=AppTest.from_file(str(Path(__file__).resolve().parents[1]/'app.py')).run(timeout=60)
        self.assertEqual(list(at.exception),[])
        headings=[h.value for h in at.subheader]
        self.assertIn('Next-draw probability forecast',headings)
        frames=[frame.value for frame in at.dataframe]
        self.assertTrue(any('P(1st slot)' in frame.columns for frame in frames))
