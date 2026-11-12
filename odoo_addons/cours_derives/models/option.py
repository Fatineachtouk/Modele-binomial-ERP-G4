import math
from odoo import api, fields, models
from odoo.exceptions import UserError

class OptionContract(models.Model):
    _name = "cours.option"
    _description = "Option vanille"

    name = fields.Char(string="Reference", required=True)
    option_type = fields.Selection([("call", "Call"), ("put", "Put")],
                                  string="Type", required=True, default="call")
    S_T = fields.Float(string="Sous-jacent a l'echeance")
    K = fields.Float(string="Strike")
    premium = fields.Float(string="Prime")
    payoff = fields.Float(string="Payoff", compute="_compute_payoff", store=True)
    profit = fields.Float(string="Profit net", compute="_compute_payoff", store=True)

    # Bonus Volatilite
    prices = fields.Text(string="Cours historiques")
    historical_volatility = fields.Float(string="Volatilite annualisee", readonly=True)

    @api.depends("option_type", "S_T", "K", "premium")
    def _compute_payoff(self):
        for rec in self:
            if rec.option_type == "call":
                rec.payoff = max(rec.S_T - rec.K, 0.0)
            else:
                rec.payoff = max(rec.K - rec.S_T, 0.0)
            rec.profit = rec.payoff - rec.premium

    def put_par_parite(self, C, S0, K, r, T):
        return C - S0 + K * math.exp(-r * T)

    def action_estimate_volatility(self):
        for rec in self:
            if not rec.prices:
                raise UserError("Renseigner les cours.")
            price_list = [float(p.strip()) for p in rec.prices.split(",") if p.strip()]
            returns = [math.log(price_list[i] / price_list[i - 1]) for i in range(1, len(price_list))]
            mean = sum(returns) / len(returns)
            variance = sum((x - mean) ** 2 for x in returns) / len(returns)
            rec.historical_volatility = math.sqrt(variance) * math.sqrt(252)