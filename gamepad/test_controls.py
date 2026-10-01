"""Input regressions: run with python -m unittest discover -s MuJoCo/gamepad."""
import math
import unittest

from controls import AXES, BUTTONS, Gate, radial, valid


def sample(**changes):
    row = dict(connected=True, device='test-controller')
    row.update({k: 0. for k in AXES})
    row.update({k: False for k in BUTTONS})
    row.update(changes)
    return row


class InputTests(unittest.TestCase):
    def setUp(self):
        self.gate = Gate('lab')
        self.update = lambda row=None, **kw: self.gate.update(
            sample() if row is None else row, kw.get('received', 10.), kw.get('now', 10.),
            kw.get('focused', True), kw.get('stop', False))
        self.update()

    def test_deadzone_and_analog_not_digital(self):
        self.assertEqual(radial(.1, .1), (0, 0))
        x, y = radial(1, 1)
        self.assertAlmostEqual(math.hypot(x, y), 1)
        full = self.update(sample(rt=1., ly=1.)).linear[0]
        half = self.update(sample(rt=1., ly=.5)).linear[0]
        self.assertGreater(half, 0)
        self.assertLess(half, full)

    def test_requires_trigger(self):
        self.assertFalse(self.update(sample(ly=1.)).enabled)
        self.assertTrue(self.update(sample(rt=1., ly=1.)).enabled)
        self.assertFalse(self.update(sample(ly=1.)).enabled)

    def test_finite_and_range_validation(self):
        for value in (float('nan'), float('inf'), 1.01, -1.01, '0', None):
            with self.subTest(value=value):
                row = sample(lx=value)
                self.assertFalse(valid(row))
                self.assertFalse(self.update(row).enabled)
        self.assertFalse(valid(sample(lt=-.1)))
        self.assertFalse(valid(sample(a=1)))

    def test_timeout_rearms_only_at_neutral(self):
        self.assertFalse(self.update(sample(rt=1.), now=10.3).enabled)
        self.assertFalse(self.update(sample(rt=1., ly=1.)).enabled)
        self.update()
        self.assertTrue(self.update(sample(rt=1., ly=1.)).enabled)

    def test_focus_loss_clears_permission(self):
        self.assertFalse(self.update(sample(rt=1., ly=1.), focused=False).enabled)
        self.assertFalse(self.update(sample(rt=1., ly=1.)).enabled)
        self.update()
        self.assertTrue(self.update(sample(rt=1., ly=1.)).enabled)

    def test_disconnect_or_new_device_requires_rearm(self):
        self.assertFalse(self.update(dict(connected=False)).enabled)
        self.assertFalse(self.update(sample(rt=1.)).enabled)
        self.update()
        self.assertFalse(self.update(sample(rt=1., device='another')).enabled)
        self.update(sample(device='another'))
        self.assertTrue(self.update(sample(rt=1., device='another')).enabled)

    def test_mode_switch_while_trigger_held(self):
        i = self.update(sample(a=True, rt=1., ly=1.))
        self.assertEqual(i.mode, 'left_arm')
        self.assertFalse(i.enabled)
        self.assertFalse(self.update(sample(rt=1., ly=1.)).enabled)
        self.update()
        self.assertTrue(self.update(sample(rt=1., ly=1.)).enabled)

    def test_lab_fixed_axes_and_upstream_modes(self):
        for key in ('x', 'y'):
            self.assertFalse(self.update(sample(**{key: True})).enabled)
            self.assertEqual(self.gate.mode, 'right_arm')
            self.update()
        g = Gate('upstream')
        for key, mode in (('x', 'chassis'), ('y', 'lift')):
            g.update(sample(), 0., 0., True)
            self.assertEqual(g.update(sample(**{key: True}), 0., 0., True).mode, mode)

    def test_pause_and_resume_latched_and_debounced(self):
        i = self.update(sample(menu=True, rt=1.))
        self.assertIn('PAUSED', i.reason)
        self.assertIn('PAUSED', self.update(sample(menu=True)).reason)
        self.update()
        self.assertFalse(self.update(sample(menu=True)).enabled)
        self.update()
        self.assertTrue(self.update(sample(rt=1.)).enabled)
        self.assertFalse(self.update(sample(rt=1.), stop=True).enabled)
        self.assertIn('PAUSED', self.update().reason)

    def test_axes_signs_precision_and_bumpers(self):
        i = self.update(sample(rt=1., lx=1., ry=1., rx=-1., lt=1., lb=True))
        self.assertLess(i.linear[1], 0)
        self.assertGreater(i.linear[2], 0)
        self.assertGreater(i.angular[2], 0)
        self.assertEqual(i.precision, .25)
        self.assertEqual(i.gripper, 1)
        self.assertEqual(self.update(sample(rt=1., rb=True)).gripper, -1)
        self.assertEqual(self.update(sample(rt=1., rb=True, lb=True)).gripper, 0)


if __name__ == '__main__':
    unittest.main()
