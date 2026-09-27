import decimal as dc
import random
import math
import matplotlib.pyplot as plt
import numpy as np

D = dc.Decimal


class ChaoticMap:
    # Constantes exatas (parsing de string não depende de dc.getcontext().prec) —
    # seguras para cachear em nível de classe.
    _ZERO = D('0')
    _ONE = D('1')
    _TWO = D('2')
    _FOUR = D('4')
    _NEG_ONE = D('-1')
    _NEG_TWO = D('-2')
    _NEG_FOUR = D('-4')
    _PI = D(str(math.pi))

    # Override para True em mapas cuja derivada é sempre positiva
    # (evita abs() por iteração): Bernoulli, GeneralizedBernoulli, KT1, KT2.
    _deriv_nonneg = False

    def __init__(self, steps, trans, x0=None, prec=50, seed=None):
        dc.getcontext().prec = prec
        self.steps = steps
        self.trans = trans
        self.prec = prec
        self._orbit_cache = None

        if seed is not None:
            random.seed(seed)

        self.x0 = self._to_decimal(x0) if x0 is not None else self.__get_initial_condition()

    def __repr__(self):
        return f"<{self.__class__.__name__}: x0={self.x0:.4f}, steps={self.steps}, trans={self.trans}>"

    @staticmethod
    def _to_decimal(v):
        # D(int) evita o parsing de string quando o valor já é int.
        return D(v) if isinstance(v, int) else D(str(v))

    def _local_ctx(self):
        # Isola dc.getcontext().prec por chamada, evitando contaminação
        # entre instâncias com prec diferente.
        ctx = dc.getcontext().copy()
        ctx.prec = self.prec
        return dc.localcontext(ctx)

    def __get_initial_condition(self):
        a, b = self.domain
        return D(str(random.uniform(a, b)))

    def _get_orbit(self):
        # Órbita (pós-transiente) computada uma única vez e reaproveitada por
        # time_series, lyapunov_estimated, lyapunov_convergence, return_map
        # e plot_density (via time_series).
        if self._orbit_cache is None:
            with self._local_ctx():
                f = self.f
                x = self.x0
                for _ in range(self.trans):
                    x = f(x)
                orbit = []
                for _ in range(self.steps):
                    x = f(x)
                    orbit.append(x)
                self._orbit_cache = orbit
        return self._orbit_cache

    def lyapunov_estimated(self, dec=False):
        orbit = self._get_orbit()
        with self._local_ctx():
            df = self.df
            soma = D(0)
            if self._deriv_nonneg:
                for x in orbit:
                    deriv = df(x)
                    soma += deriv.ln() if deriv > 0 else D("-1e10")
            else:
                for x in orbit:
                    deriv = abs(df(x))
                    soma += deriv.ln() if deriv > 0 else D("-1e10")
            lambda_est = soma / D(self.steps)
        return lambda_est if dec else float(lambda_est)

    def lyapunov_convergence(self, plot=False):
        orbit = self._get_orbit()
        with self._local_ctx():
            df = self.df
            soma = D(0)
            evolution = []
            if self._deriv_nonneg:
                for i, x in enumerate(orbit, start=1):
                    soma += df(x).ln()
                    evolution.append(soma / D(i))
            else:
                for i, x in enumerate(orbit, start=1):
                    soma += abs(df(x)).ln()
                    evolution.append(soma / D(i))

        if plot:
            self._plot_convergence(evolution)

        return np.array([float(v) for v in evolution])

    def _plot_convergence(self, data):
        plt.figure(figsize=(10, 4))
        plt.plot(data, lw=0.5, color='#2c3e50', label="Estimated $\\lambda$")
        plt.axhline(float(self.theoretical_lyapunov), color='r', ls='--', lw=0.8, label="Theoretical")
        plt.title(f"Lyapunov Convergence: {self.__class__.__name__} (x0={float(self.x0):.4f})")
        plt.xlabel("n (iterations)")
        plt.ylabel("$\\lambda$")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    def time_series(self, dec=False, plot=False):
        orbit = self._get_orbit()

        if plot:
            self._plot_series(orbit)

        return orbit if dec else np.array(orbit, dtype=float)

    def _plot_series(self, data):
        plt.figure(figsize=(10, 4))
        plt.plot(data, lw=0.5, color='#2c3e50')
        plt.title(f"Time Series: {self.__class__.__name__} (x0={float(self.x0):.4f})")
        plt.xlabel("n (iterations)")
        plt.ylabel("$x_n$")
        plt.grid(True, alpha=0.3)
        plt.show()

    def lyapunov_summary(self, dec=False):
        est = self.lyapunov_estimated(dec=dec)
        series = self.time_series(dec=dec)

        theo = None
        error_str = "N/A"

        try:
            theo_val = self.theoretical_lyapunov
            if theo_val is not None:
                theo = theo_val if dec else float(theo_val)
                theo_dec = D(str(theo_val))
                est_dec = D(str(est))

                error = abs((est_dec - theo_dec) / theo_dec) * 100 if theo_dec != 0 else D(0)
                error_str = f"{error:.8f}%" if dec else f"{float(error):.4f}%"
        except (NotImplementedError, TypeError):
            pass

        return {
            "map": self.__class__.__name__,
            "theoretical": theo,
            "estimated": est,
            "error_percent": error_str,
            "steps": self.steps,
            "transient": self.trans,
            "x0": self.x0 if dec else float(self.x0),
            "time_series": series
        }

    def plot_density(self, bins=50, figsize=(8, 5), color='#2c3e50', show_analytical=True, style='scatter'):
        orbit = self.time_series()

        a, b = float(self.domain[0]), float(self.domain[1])
        contagens, bordas = np.histogram(orbit, bins=bins, range=(a, b))
        delta_x = bordas[1] - bordas[0]
        rho_est = contagens / (len(orbit) * delta_x)
        x_bins = 0.5 * (bordas[:-1] + bordas[1:])

        fig, ax = plt.subplots(figsize=figsize)

        if style == 'scatter':
            ax.scatter(x_bins, rho_est, s=10, color=color, alpha=0.7, label=r'$\rho_{\mathrm{est}}$')
        elif style == 'bar':
            ax.bar(x_bins, rho_est, width=delta_x, alpha=0.6, color=color, edgecolor='k', linewidth=0.4, label=r'$\rho_{\mathrm{est}}$')
        else:
            raise ValueError(f"style='{style}' inválido. Use 'scatter' ou 'bar'.")

        if show_analytical:
            try:
                x_curve = np.linspace(a + 1e-9, b - 1e-9, 500)
                # Curva de referência: prec reduzido (não precisa do prec da instância
                # para uma linha de plot; vetorização real em float64 exigiria
                # reimplementar cada density() fora do Decimal).
                with dc.localcontext() as ctx:
                    ctx.prec = 20
                    rho_vals = np.array([float(self.density(D(str(xi)))) for xi in x_curve])
                ax.plot(x_curve, rho_vals, color='cornflowerblue', linewidth=2, label=r'$\rho_{\mathrm{exato}}$')
            except (NotImplementedError, AttributeError, ZeroDivisionError):
                pass

        ax.set_ylim(0, np.percentile(rho_est, 99) * 1.3)
        ax.set_xlabel('$x$')
        ax.set_ylabel(r'$\rho(x)$')
        ax.set_title(f'Invariant Density — {self.__class__.__name__}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def return_map(self, plot=False):
        orbit = self._get_orbit()
        x_n = orbit[:-1]
        x_n1 = orbit[1:]

        if plot:
            self._plot_return_map(x_n, x_n1)

        return x_n, x_n1

    def _plot_return_map(self, x_n, x_n1):
        plt.figure(figsize=(6, 6))
        plt.plot(x_n, x_n1, ',', color='#2c3e50', alpha=0.4, markersize=1.5)
        plt.title(f"Return Map — {self.__class__.__name__} (x0={float(self.x0):.4f})")
        plt.xlabel("$x_n$")
        plt.ylabel("$x_{n+1}$")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


# ============== Maps ==========================================================================================

class LogisticMap(ChaoticMap):
    domain = (0, 1)

    def __init__(self, steps, trans, r=4, x0=None, prec=50, seed=None):
        self.r = self._to_decimal(r)
        super().__init__(steps, trans, x0, prec, seed)

    def f(self, x):
        return self.r * x * (self._ONE - x)

    def df(self, x):
        return self.r * (self._ONE - self._TWO * x)

    @property
    def theoretical_lyapunov(self):
        if self.r == self._FOUR:
            return self._TWO.ln()
        return None

    def density(self, x):
        denom = self._PI * (x * (self._ONE - x)).sqrt()
        return self._ONE / denom if denom != 0 else D('inf')


class GeneralizedLogisticMap(ChaoticMap):
    domain = (0, 1)

    def __init__(self, steps, trans, b=1, x0=None, prec=50, seed=None):
        self.b = self._to_decimal(b)
        super().__init__(steps, trans, x0, prec, seed)

    def f(self, x):
        return (self._FOUR ** self.b) * x * (self._ONE - x ** (self._ONE / self.b)) ** self.b

    def df(self, x):
        term = self._ONE - x ** (self._ONE / self.b)
        return (self._FOUR ** self.b) * (term ** (self.b - self._ONE)) * (self._ONE - self._TWO * x ** (self._ONE / self.b))

    @property
    def theoretical_lyapunov(self):
        return self._TWO.ln()

    def density(self, x):
        exp = self._ONE / (self._TWO * self.b) - self._ONE
        numerator = x ** exp
        denominator = (self._ONE - x ** (self._ONE / self.b)).sqrt()
        return numerator / (self._PI * self.b * denominator)


class UlamMap(ChaoticMap):
    domain = (-1, 1)

    def f(self, x):
        return self._ONE - self._TWO * x * x

    def df(self, x):
        return self._NEG_FOUR * x

    @property
    def theoretical_lyapunov(self):
        return self._TWO.ln()

    def density(self, x):
        return self._ONE / (self._PI * (self._ONE - x * x).sqrt())


class GeneralizedUlamMap(ChaoticMap):
    domain = (-1, 1)

    def __init__(self, steps, trans, r, x0=None, prec=50, seed=None):
        self.r = self._to_decimal(r)
        super().__init__(steps, trans, x0, prec, seed)

    def f(self, x):
        return self._ONE - self.r * x * x

    def df(self, x):
        return self._NEG_TWO * self.r * x

    @property
    def theoretical_lyapunov(self):
        raise NotImplementedError("No closed form available")


class BernoulliMap(ChaoticMap):
    domain = (0, 1)
    _deriv_nonneg = True

    def f(self, x):
        return (self._TWO * x) % self._ONE

    def df(self, x):
        return self._TWO

    @property
    def theoretical_lyapunov(self):
        return self._TWO.ln()

    def density(self, x):
        return self._ONE


class GaussMap(ChaoticMap):
    domain = (1e-12, 0.999999999999)

    def f(self, x):
        if x == 0:
            return self._ZERO
        inv_x = self._ONE / x
        return inv_x - inv_x.to_integral_value(rounding=dc.ROUND_FLOOR)

    def df(self, x):
        if x == 0:
            return self._ZERO
        return self._NEG_ONE / (x * x)

    @property
    def theoretical_lyapunov(self):
        return (self._PI * self._PI) / (D('6') * self._TWO.ln())

    def density(self, x):
        ln2 = self._TWO.ln()
        return self._ONE / (ln2 * (self._ONE + x))


class TentMap(ChaoticMap):
    domain = (0, 1)

    def f(self, x):
        return self._TWO * min(x, self._ONE - x)

    def df(self, x):
        return self._TWO if x < D('0.5') else self._NEG_TWO

    @property
    def theoretical_lyapunov(self):
        return self._TWO.ln()

    def density(self, x):
        return self._ONE


class AsymetricMap(ChaoticMap):
    domain = (0, 1)
    _A = D('0.4')
    _B = D('0.6')

    def f(self, x):
        return (x / self._A) if x < self._A else ((self._ONE - x) / self._B)

    def df(self, x):
        return self._ONE / self._A if x < self._A else self._NEG_ONE / self._B

    @property
    def theoretical_lyapunov(self):
        a = self._A
        return -(a * a.ln()) - (self._B * self._B.ln())

    def density(self, x):
        return self._ONE


class ChebyshevMap(ChaoticMap):
    domain = (-1, 1)

    def __init__(self, steps, trans, k=2, x0=None, prec=50, seed=None):
        self.k = int(k)
        super().__init__(steps, trans, x0, prec, seed)

    def f(self, x):
        t0, t1 = self._ONE, x
        for _ in range(self.k - 1):
            t0, t1 = t1, self._TWO * x * t1 - t0
        return t1

    def df(self, x):
        u0, u1 = self._ONE, self._TWO * x
        for _ in range(self.k - 2):
            u0, u1 = u1, self._TWO * x * u1 - u0
        return D(self.k) * u1

    @property
    def theoretical_lyapunov(self):
        return D(self.k).ln()

    def density(self, x):
        return self._ONE / (self._PI * (self._ONE - x * x).sqrt())


class GeneralizedBernoulliMap(ChaoticMap):
    domain = (0, 1)
    _deriv_nonneg = True

    def __init__(self, steps, trans, m=2, x0=None, prec=50, seed=None):
        self.m = self._to_decimal(m)
        super().__init__(steps, trans, x0, prec, seed)

    def f(self, x):
        return (self.m * x) % self._ONE

    def df(self, x):
        return self.m

    @property
    def theoretical_lyapunov(self):
        return self.m.ln()

    def density(self, x):
        return self._ONE


class KT1Map(ChaoticMap):
    domain = (0, 1)
    _deriv_nonneg = True

    def __init__(self, steps, trans, gamma=None, x0=None, prec=50, seed=None):
        self.gamma = self._to_decimal(gamma)
        super().__init__(steps, trans, x0, prec, seed)

    def f(self, x):
        g = self.gamma
        if x < g:
            return x / g
        else:
            return (g * x - g * g) / (self._ONE - g)

    def df(self, x):
        g = self.gamma
        if x < g:
            return self._ONE / g
        else:
            return g / (self._ONE - g)

    @property
    def theoretical_lyapunov(self):
        g = self.gamma
        term1 = (self._ONE / (self._TWO - g)) * (self._ONE / g).ln()
        term2 = ((self._ONE - g) / (self._TWO - g)) * (g / (self._ONE - g)).ln()
        return term1 + term2


class KT2Map(ChaoticMap):
    domain = (0, 1)
    _deriv_nonneg = True

    def __init__(self, steps, trans, gamma=None, x0=None, prec=50, seed=None):
        self.gamma = self._to_decimal(gamma)
        super().__init__(steps, trans, x0, prec, seed)

    def f(self, x):
        g = self.gamma
        if x < self._ONE - g:
            return (g * x) / (self._ONE - g) + self._ONE - g
        else:
            return (x - (self._ONE - g)) / g

    def df(self, x):
        g = self.gamma
        if x < self._ONE - g:
            return g / (self._ONE - g)
        else:
            return self._ONE / g

    @property
    def theoretical_lyapunov(self):
        g = self.gamma
        term1 = (self._ONE / (self._TWO - g)) * (self._ONE / g).ln()
        term2 = ((self._ONE - g) / (self._TWO - g)) * (g / (self._ONE - g)).ln()
        return term1 + term2


class Manneville(ChaoticMap):
    domain = (0, 1)

    def __init__(self, steps, trans, epslon=None, x0=None, prec=50, seed=None):
        self.epslon = self._to_decimal(epslon)
        super().__init__(steps, trans, x0, prec, seed)

    def f(self, x):
        e = self.epslon
        return ((self._ONE + e) * x + (self._ONE - e) * x * x) % self._ONE

    def df(self, x):
        e = self.epslon
        return (self._ONE + e) + self._TWO * (self._ONE - e) * x

    @property
    def theoretical_lyapunov(self):
        raise NotImplementedError("No closed form available")

    def density(self, x):
        e = self.epslon
        K = (self._ONE - e) / ((self._TWO - e).ln() - e.ln())
        term1 = self._ONE / (e + (self._ONE - e) * x)
        term2 = self._ONE / (self._ONE + (self._ONE - e) * x)
        return K * (term1 + term2)


class ConjugateTentMap(ChaoticMap):
    domain = (0, 1)

    def __init__(self, steps, trans, p, x0=None, prec=50, seed=None):
        self.p = self._to_decimal(p)
        super().__init__(steps, trans, x0, prec, seed)

    def f(self, x):
        p = self.p
        q = self._ONE - p
        if x <= p * p:
            return x / (p * p)
        else:
            t = self._ONE - x.sqrt()
            return t * t / (q * q)

    def df(self, x):
        p = self.p
        q = self._ONE - p
        if x <= p * p:
            return self._ONE / (p * p)
        else:
            sx = x.sqrt()  # calculado uma única vez (original chamava .sqrt() 2x)
            return -(self._ONE - sx) / (q * q * sx)

    @property
    def theoretical_lyapunov(self):
        raise NotImplementedError("No closed form available")

    def density(self, x):
        return self._ONE / (self._TWO * x.sqrt())


class ThalerMap(ChaoticMap):
    domain = (0, 1)

    def __init__(self, steps, trans, z, x0=None, prec=50, seed=None):
        self.z = self._to_decimal(z)
        super().__init__(steps, trans, x0, prec, seed)

    def f(self, x):
        z = self.z
        inner = (x / (self._ONE + x)) ** (z - self._TWO) - x ** (z - self._TWO)
        return (x * (self._ONE + inner) ** (self._NEG_ONE / (z - self._TWO))) % self._ONE

    def df(self, x):
        z = self.z
        a = z - self._TWO
        ratio = x / (self._ONE + x)
        g = self._ONE + ratio ** a - x ** a
        dg = (z - self._TWO) * (ratio ** (a - self._ONE) / (self._ONE + x) ** 2 - x ** (a - self._ONE))
        exp = self._NEG_ONE / a

        return g ** exp - x * g ** (exp - self._ONE) * dg

    @property
    def theoretical_lyapunov(self):
        raise NotImplementedError("No closed form available")

    def density(self, x):
        z = self.z
        alpha = self._ONE / (z - self._ONE)
        exp = self._NEG_ONE / alpha
        return x ** exp + (self._ONE + x) ** exp
