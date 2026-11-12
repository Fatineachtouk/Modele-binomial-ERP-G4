from odoo.tests import TransactionCase, tagged
from odoo.exceptions import ValidationError

@tagged("post_install", "-at_install")
class TestDerives(TransactionCase):

    def test_forward_price(self):
        fwd = self.env["cours.forward"].create({
            "name": "T1", "S0": 80.0, "r": 0.04, "T": 0.5
        })
        self.assertAlmostEqual(fwd.F0, 81.62, places=2)

    def test_call_payoff(self):
        opt = self.env["cours.option"].create({
            "name": "T2", "option_type": "call", "K": 50.0, "S_T": 45.0
        })
        self.assertEqual(opt.payoff, 0.0)

    def test_forward_value(self):
        fwd = self.env["cours.forward"].create({
            "name": "TFwdA", "S0": 100.0, "r": 0.04, "T": 1.0,
            "F0": 100.0, "t_elapsed": 0.5, "S_t": 108.0
        })
        self.assertAlmostEqual(fwd.value, 9.98, places=2)

    def test_arbitrage_gain(self):
        fwd = self.env["cours.forward"].create({
            "name": "TFwdB", "S0": 80.0, "r": 0.04, "T": 0.5, "market_price": 82.0
        })
        self.assertAlmostEqual(fwd.arbitrage_gain, 0.38, places=2)
        self.assertEqual(fwd.strategy, "sell")

    def test_portfolio(self):
        port = self.env["cours.portfolio"].create({
            "name": "Port1", "mu_A": 0.12, "sigma_A": 0.20,
            "mu_B": 0.06, "sigma_B": 0.10, "rho": 0.5, "w_A": 0.6
        })
        self.assertAlmostEqual(port.exp_return, 0.096, places=3)
        self.assertAlmostEqual(port.sigma_p, 0.1442, places=4)