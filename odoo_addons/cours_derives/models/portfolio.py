import math
from odoo import api, fields, models
from odoo.exceptions import ValidationError

class Portfolio(models.Model):
    _name = "cours.portfolio"
    _description = "Portefeuille de deux actifs"

    name = fields.Char(string="Nom du portefeuille", required=True)
    mu_A = fields.Float(string="mu_A", required=True)
    sigma_A = fields.Float(string="sigma_A", required=True)
    mu_B = fields.Float(string="mu_B", required=True)
    sigma_B = fields.Float(string="sigma_B", required=True)
    rho = fields.Float(string="rho", required=True)
    w_A = fields.Float(string="w_A", required=True)

    w_B = fields.Float(string="w_B", compute="_compute_portfolio", store=True)
    exp_return = fields.Float(string="E(Rp)", compute="_compute_portfolio", store=True)
    sigma_p = fields.Float(string="sigma_p", compute="_compute_portfolio", store=True)

    @api.constrains("rho", "w_A")
    def _check_parameters(self):
        for rec in self:
            if not (-1.0 <= rec.rho <= 1.0):
                raise ValidationError("rho doit etre entre -1 et 1.")
            if not (0.0 <= rec.w_A <= 1.0):
                raise ValidationError("w_A doit etre entre 0 et 1.")

    @api.depends("mu_A", "sigma_A", "mu_B", "sigma_B", "rho", "w_A")
    def _compute_portfolio(self):
        for rec in self:
            rec.w_B = 1.0 - rec.w_A
            rec.exp_return = (rec.w_A * rec.mu_A) + (rec.w_B * rec.mu_B)
            var = (
                (rec.w_A ** 2) * (rec.sigma_A ** 2) +
                (rec.w_B ** 2) * (rec.sigma_B ** 2) +
                2.0 * rec.w_A * rec.w_B * rec.rho * rec.sigma_A * rec.sigma_B
            )
            rec.sigma_p = math.sqrt(max(var, 0.0))