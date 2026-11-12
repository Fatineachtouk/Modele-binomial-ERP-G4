import math
from odoo import api, fields, models
from odoo.exceptions import ValidationError

class ForwardContract(models.Model):
    _name = "cours.forward"
    _description = "Contrat forward"

    TOL = 0.01

    name = fields.Char(string="Reference", required=True)
    S0 = fields.Float(string="Prix spot", required=True)
    r = fields.Float(string="Taux sans risque (decimal)", required=True)
    T = fields.Float(string="Maturite (ans)", required=True)
    F0 = fields.Float(
        string="Prix forward",
        digits=(16, 4),
        compute="_compute_F0",
        store=True,
        readonly=False
    )

    # Exercice A : En cours de vie
    t_elapsed = fields.Float(string="Temps ecoule (ans)", default=0.0)
    S_t = fields.Float(string="Prix spot courant", default=0.0)
    value = fields.Float(string="Valeur du forward (Vt)", compute="_compute_value", store=True)

    # Exercice B : Arbitrage
    market_price = fields.Float(string="Prix cote (marche)", default=0.0)
    arbitrage_gain = fields.Float(string="Gain d'arbitrage", compute="_compute_arbitrage", store=True)
    strategy = fields.Selection([
        ("sell", "Vendre le forward"),
        ("buy", "Acheter le forward"),
        ("none", "Aucun arbitrage")
    ], string="Strategie", compute="_compute_arbitrage", store=True)

    @api.depends("S0", "r", "T")
    def _compute_F0(self):
        for rec in self:
            rec.F0 = rec.S0 * math.exp(rec.r * rec.T)

    @api.depends("S_t", "F0", "r", "T", "t_elapsed")
    def _compute_value(self):
        for rec in self:
            rec.value = rec.S_t - rec.F0 * math.exp(-rec.r * (rec.T - rec.t_elapsed))

    @api.constrains("t_elapsed", "T")
    def _check_time(self):
        for rec in self:
            if rec.t_elapsed < 0 or rec.t_elapsed > rec.T:
                raise ValidationError("Le temps ecoule t doit etre compris entre 0 et T (0 <= t <= T) !")

    @api.depends("market_price", "F0")
    def _compute_arbitrage(self):
        for rec in self:
            if not rec.market_price:
                rec.arbitrage_gain = 0.0
                rec.strategy = "none"
            else:
                gain = rec.market_price - rec.F0
                rec.arbitrage_gain = gain
                if gain > self.TOL:
                    rec.strategy = "sell"
                elif gain < -self.TOL:
                    rec.strategy = "buy"
                else:
                    rec.strategy = "none"