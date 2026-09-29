"""Run run.py first. Tests use native model with generated mapping and tables."""
from copy import deepcopy
import os
from pathlib import Path
import re
import subprocess
import tempfile
import unittest
import yaml

HERE = Path(__file__).resolve().parent
BUILD = HERE.parents[3] / 'accelergy-timeloop-infrastructure/src/timeloop/build'


class ConnectedVPUTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = yaml.safe_load((HERE / 'outputs/parsed-processed-input.yaml').read_text())
        for table in ('ERT', 'ART'):
            cls.data.update(yaml.safe_load((HERE / f'outputs/timeloop-model.{table}.yaml').read_text()))

    def run_model(self, change=lambda d: None, valid=True):
        data = deepcopy(self.data)
        change(data)
        with tempfile.TemporaryDirectory() as tmp:
            inp = Path(tmp) / 'input.yaml'
            inp.write_text(yaml.safe_dump(data))
            env = os.environ.copy()
            env['LD_LIBRARY_PATH'] = str(BUILD) + ':' + env.get('LD_LIBRARY_PATH', '')
            p = subprocess.run([str(BUILD / 'timeloop-model'), str(inp)], cwd=tmp,
                               env=env, text=True, capture_output=True)
            report = Path(tmp) / 'timeloop-model.stats.txt'
            if not valid:
                self.assertTrue(p.returncode != 0 or not report.exists(), p.stdout + p.stderr)
                return
            self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
            self.assertTrue(report.exists(), p.stdout + p.stderr)
            text = report.read_text()
            cycles = int(re.search(r'^Cycles: (\d+)', text, re.M)[1])
            return cycles, text

    @staticmethod
    def vpu(data):
        def find(node):
            if isinstance(node, dict):
                if node.get('subclass') == 'vpu':
                    return node['attributes']
                for value in node.values():
                    result = find(value)
                    if result is not None:
                        return result
            elif isinstance(node, list):
                for value in node:
                    result = find(value)
                    if result is not None:
                        return result
        return find(data['architecture'])

    def test_hand_calculated_serial_flow(self):
        cycles, text = self.run_model()
        # Two pieces. Totals: input fetch 8, producer reads 44, MAC 64,
        # producer writes 16, VPU reads 4, ReLU 4, final DRAM write 4.
        self.assertEqual(cycles, 144)
        self.assertIn('Completed tiles: 2', text)
        self.assertIn('Output elements per tile: 8', text)
        self.assertIn('VPU compute cycles: 4', text)
        self.assertIn('Final DRAM write cycles: 4', text)

    def test_latency_interval_width(self):
        def timing(latency, interval, width=4):
            def change(data):
                a = self.vpu(data)
                a['vector_width'] = width
                a['operations']['relu'] = dict(latency=latency, initiation_interval=interval)
            return change
        self.assertEqual(self.run_model(timing(4,4))[0], 156)
        self.assertEqual(self.run_model(timing(4,1))[0], 150)
        self.assertEqual(self.run_model(timing(1,1,8))[0], 142)

    def test_disabled(self):
        self.assertEqual(self.run_model(lambda d: d['problem'].pop('vpu_stages'))[0], 64)

    def test_reject_invalid_requests(self):
        self.run_model(lambda d: d['problem']['vpu_stages'][0].update(unit='missing'), valid=False)
        self.run_model(lambda d: d['problem']['vpu_stages'][0].update(operation='mul'), valid=False)
        self.run_model(lambda d: d['problem']['vpu_stages'][0].update(operation='softmax', vector_length=16), valid=False)
        def incomplete(data):
            data.pop('architecture_constraints', None)
            for m in data['mapping']:
                if m['type'] == 'temporal':
                    m['factors'] = re.sub(r'K=?[14](?= |$)', 'K=2', m['factors'])
        self.run_model(incomplete, valid=False)


if __name__ == '__main__':
    unittest.main(verbosity=2)
