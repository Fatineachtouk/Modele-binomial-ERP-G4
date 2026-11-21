import math
from odoo import api, fields, models
from odoo.exceptions import UserError

class OptionContract(models.Model):
    _name = "cours.option"
    _description = "Option vanille"

    # === CHAMPS EXISTANTS (TP 0) ===
    name = fields.Char(string="Reference", required=True)
    option_type = fields.Selection([("call", "Call"), ("put", "Put")],
                                  string="Type", required=True, default="call")
    S_T = fields.Float(string="Sous-jacent a l'echeance")
    K = fields.Float(string="Strike (K)", default=100.0, required=True)
    premium = fields.Float(string="Prime")
    payoff = fields.Float(string="Payoff", compute="_compute_payoff", store=True)
    profit = fields.Float(string="Profit net", compute="_compute_payoff", store=True)

    # Bonus Volatilite historique
    prices = fields.Text(string="Cours historiques")
    historical_volatility = fields.Float(string="Volatilite annualisee", readonly=True)

    # ========================================================
    # NOUVEAUX CHAMPS PROJET 4 - SEMAINE 2 (MODÈLE BINOMIAL CRR)
    # ========================================================
    S0 = fields.Float(string="Prix Spot (S0)", default=100.0, required=True)
    T = fields.Float(string="Maturité T (années)", default=1.0, required=True)
    r = fields.Float(string="Taux sans risque (r)", default=0.05, required=True)
    sigma = fields.Float(string="Volatilité (σ)", default=0.20, required=True)
    
    nb_pas = fields.Integer(string="Nombre de pas (N)", default=100, required=True)
    style_option = fields.Selection([
        ("europeenne", "Européenne"),
        ("americaine", "Américaine")
    ], string="Style d'exercice", default="europeenne", required=True)

    # Résultats de pricing calculés automatiquement
    prix_binomial = fields.Float(string="Prix Binomial (CRR)", compute="_compute_pricing", store=True)
    prix_bs = fields.Float(string="Prix Black-Scholes", compute="_compute_pricing", store=True)
    ecart = fields.Float(string="Écart (|Binomial - BS|)", compute="_compute_pricing", store=True)

    # === CALCULS DU TP 0 ===
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
            # Met à jour automatiquement le sigma utilisé pour le pricing
            rec.sigma = rec.historical_volatility

    # ========================================================
    # FONCTIONS DE PRICING (CRR & BLACK-SCHOLES)
    # ========================================================
    def _norm_cdf(self, x):
        """Fonction de répartition de la loi normale centrée réduite N(x)"""
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    def _bs_price(self, S0, K, T, r, sigma, option_type):
        """Formule fermée de Black-Scholes pour options européennes"""
        if S0 <= 0 or K <= 0 or T <= 0 or sigma <= 0:
            return 0.0
        d1 = (math.log(S0 / K) + (r + 0.5 * (sigma ** 2)) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        if option_type == "call":
            return S0 * self._norm_cdf(d1) - K * math.exp(-r * T) * self._norm_cdf(d2)
        else:
            return K * math.exp(-r * T) * self._norm_cdf(-d2) - S0 * self._norm_cdf(-d1)

    def _crr_price(self, S0, K, T, r, sigma, N, option_type, is_american):
        """Arbre binomial de Cox-Ross-Rubinstein (Européen et Américain)"""
        if S0 <= 0 or K <= 0 or T <= 0 or sigma <= 0 or N <= 0:
            return 0.0
        
        dt = T / N
        u = math.exp(sigma * math.sqrt(dt))
        d = 1.0 / u
        p = (math.exp(r * dt) - d) / (u - d)
        
        # Sécurité numérique sur la probabilité risque-neutre
        p = max(0.0, min(1.0, p))
        disc = math.exp(-r * dt)

        # Nœuds terminaux à l'échéance T
        ST = [S0 * (u ** j) * (d ** (N - j)) for j in range(N + 1)]
        if option_type == "call":
            V = [max(s - K, 0.0) for s in ST]
        else:
            V = [max(K - s, 0.0) for s in ST]

        # Induction rétrograde de N-1 à 0
        for i in range(N - 1, -1, -1):
            for j in range(i + 1):
                continuation = disc * (p * V[j + 1] + (1.0 - p) * V[j])
                if is_american:
                    S_ij = S0 * (u ** j) * (d ** (i - j))
                    payoff_immediat = max(S_ij - K, 0.0) if option_type == "call" else max(K - S_ij, 0.0)
                    V[j] = max(continuation, payoff_immediat)
                else:
                    V[j] = continuation

        return V[0]

    @api.depends("S0", "K", "T", "r", "sigma", "nb_pas", "option_type", "style_option")
    def _compute_pricing(self):
        """Calcul réactif des prix CRR, Black-Scholes et de l'écart"""
        for rec in self:
            is_american = (rec.style_option == "americaine")
            rec.prix_binomial = rec._crr_price(rec.S0, rec.K, rec.T, rec.r, rec.sigma, rec.nb_pas, rec.option_type, is_american)
            rec.prix_bs = rec._bs_price(rec.S0, rec.K, rec.T, rec.r, rec.sigma, rec.option_type)
            rec.ecart = abs(rec.prix_binomial - rec.prix_bs)