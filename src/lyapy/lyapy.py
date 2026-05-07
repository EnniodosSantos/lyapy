import decimal as dc
import random
import math
import matplotlib.pyplot as plt
import numpy as np

# Atalho interno para não repetir dc.Decimal em todo o código
D = dc.Decimal


class ChaoticMap:
    def __init__(self, steps, trans, x0=None, prec=50, seed=None):
        dc.getcontext().prec = prec
        self.steps = steps
        self.trans = trans
        self.prec = prec

        if seed is not None:
            random.seed(seed)

        self.x0 = D(str(x0)) if x0 is not None else self.__get_initial_condition()

    def __repr__(self):
        return f"<{self.__class__.__name__}: x0={self.x0:.4f}, steps={self.steps}, trans={self.trans}>"

    def __get_initial_condition(self):
        a, b = self.domain
        return D(str(random.uniform(a, b)))

    def lyapunov_estimated(self, dec=False):
        x = self.x0
        soma = D(0)

        for _ in range(self.trans):
            x = self.f(x)

        for _ in range(self.steps):
            x = self.f(x)
            deriv = abs(self.df(x))
            if deriv > 0:
                soma += deriv.ln()
            else:
                soma += D("-1e10")

        lambda_est = soma / D(self.steps)
        return lambda_est if dec else float(lambda_est)

    def lyapunov_convergence(self, plot=False):
        x = self.x0
        for _ in range(self.trans):
            x = self.f(x)

        soma = D(0)
        evolution = []

        for i in range(1, self.steps + 1):
            x = self.f(x)
            soma += abs(self.df(x)).ln()
            evolution.append(soma / D(i))

        if plot:
            self._plot_convergence(evolution)

        return np.array([float(v) for v in evolution])

    def _plot_convergence(self, data):
        plt.figure(figsize=(8, 4))
        plt.plot(data, label="Estimated $\lambda$")
        plt.axhline(float(self.theoretical_lyapunov), color='r', ls='--', label="Theoretical")
        plt.title(f"Lyapunov Convergence - {self.__class__.__name__}")
        plt.xlabel("Iterations")
        plt.ylabel("$\lambda$")
        plt.legend()
        plt.grid(True)
        plt.show()

    def time_series(self, dec=False, plot=False):
        x = self.x0
        for _ in range(self.trans):
            x = self.f(x)

        orbit = []
        for _ in range(self.steps):
            x = self.f(x)
            orbit.append(x)

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

        # Inicializa variáveis de comparação
        theo = None
        error_str = "N/A"

        try:
            theo_val = self.theoretical_lyapunov
            if theo_val is not None:
                theo = theo_val if dec else float(theo_val)
                theo_dec = D(str(theo_val))
                est_dec = D(str(est))

                # Cálculo do erro com fallback para zero
                error = abs((est_dec - theo_dec) / theo_dec) * 100 if theo_dec != 0 else D(0)
                error_str = f"{error:.8f}%" if dec else f"{float(error):.4f}%"
        except (NotImplementedError, TypeError):
            # Se não houver valor teórico, theo e error_str permanecem como None e "N/A"
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
            raise ValueError(f"style='{style}'Use 'scatter' or 'bar'.")

        if show_analytical:
        try:
            x_curve = np.linspace(a + 1e-9, b - 1e-9, 500)
            rho_vals = np.array([float(self.density(D(str(xi)))) for xi in x_curve])
            ax.plot(x_curve, rho_vals, color='cornflowerblue', linewidth=2, label=r'$\rho_{\mathrm{exato}}$')
        except (NotImplementedError, AttributeError, ZeroDivisionError):
            pass

        ax.set_xlabel('$x$')
        ax.set_ylabel(r'$\rho(x)$')
        ax.set_title(f'Invariant Density — {self.__class__.__name__}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

# ============== Maps ==========================================================================================

class LogisticMap(ChaoticMap):
    domain = (0, 1)

    def __init__(self, steps, trans, r=4, x0=None, prec=50, seed=None):
        self.r = D(str(r))
        super().__init__(steps, trans, x0, prec, seed)

    def f(self, x):
        return self.r * x * (D('1') - x)

    def df(self, x):
        return self.r * (D('1') - D('2') * x)

    @property
    def theoretical_lyapunov(self):
        if self.r == D('4'):
            return D('2').ln()
        return None

    def density(self, x):
        pi = D(str(math.pi))
        denom = pi * (x * (D('1') - x)).sqrt()
        return D('1') / denom if denom != 0 else D('inf')


class GeneralizedLogisticMap(ChaoticMap):
    domain = (0, 1)

    def __init__(self, steps, trans, b=1, x0=None, prec=50, seed=None):
        self.b = D(str(b))          # era self.r = D(str(r)) — variável errada
        super().__init__(steps, trans, x0, prec, seed)

    def f(self, x):
        return (D('4') ** self.b) * x * (D('1') - x ** (D('1') / self.b)) ** self.b

    def df(self, x):
        term = D('1') - x ** (D('1') / self.b)
        return (D('4') ** self.b) * (term ** (self.b - D('1'))) * (D('1') - D('2') * x ** (D('1') / self.b))


    @property
    def theoretical_lyapunov(self):
        return D('2').ln()

    def density(self, x):
        pi = D(str(math.pi))
        exp = D('1') / (D('2') * self.b) - D('1')          # 1/(2b) - 1
        numerator = x ** exp                                 # x^(1/(2b) - 1)
        denominator = (D('1') - x ** (D('1') / self.b)).sqrt()  # sqrt(1 - x^(1/b))
        return numerator / (pi * self.b * denominator)




class UlamMap(ChaoticMap):
    domain = (-1, 1)

    def f(self, x):
        return D('1') - D('2') * x**2

    def df(self, x):
        return D('-4') * x

    @property
    def theoretical_lyapunov(self):
        return D('2').ln()

    def density(self, x):
        pi = D(str(math.pi))
        return D('1') / (pi * (D('1') - x**2).sqrt())


class GeneralizedUlamMap(ChaoticMap):
    domain = (-1,1)

    def __init__(self, steps, trans, r, x0=None, prec=50, seed=None):
        self.r = D(str(r))
        super().__init__(steps, trans, x0, prec, seed)

    def f(self, x):
        return D('1') - self.r* x**2

    def df(self, x):
        return D('-2')*self.r*x

    @property
    def theoretical_lyapunov(self):
        raise NotImplementedError("No closed form available")


class BernoulliMap(ChaoticMap):
    domain = (0, 1)

    def f(self, x):
        return (D('2') * x) % D('1')

    def df(self, x):
        return D('2')

    @property
    def theoretical_lyapunov(self):
        return D('2').ln()

    def density(self, x):
        return D('1')


class GaussMap(ChaoticMap):
    domain = (1e-12, 0.999999999999)

    def f(self, x):
        if x == 0:
            return D(0)
        inv_x = D('1') / x
        return inv_x - inv_x.to_integral_value(rounding=dc.ROUND_FLOOR)

    def df(self, x):
        if x == 0:
            return D(0)
        return D('-1') / (x**2)

    @property
    def theoretical_lyapunov(self):
        pi = D(str(math.pi))
        return (pi**2) / (D('6') * D('2').ln())

    def density(self, x):
        ln2 = D('2').ln()
        return D('1') / (ln2 * (D('1') + x))


class TentMap(ChaoticMap):
    domain = (0, 1)

    def f(self, x):
        return D('2') * min(x, D('1') - x)

    def df(self, x):
        return D('2') if x < D('0.5') else D('-2')

    @property
    def theoretical_lyapunov(self):
        return D('2').ln()

    def density(self, x):
        return D('1')


class AsymetricMap(ChaoticMap):
    domain = (0, 1)

    def f(self, x):
        return (x / D('0.4')) if x < D('0.4') else ((D('1') - x) / D('0.6'))

    def df(self, x):
        return D('1') / D('0.4') if x < D('0.4') else D('-1') / D('0.6')

    @property
    def theoretical_lyapunov(self):
        a = D('0.4')
        return -(a * a.ln()) - ((D('1') - a) * (D('1') - a).ln())

    def density(self, x):
        return D('1')


class ChebyshevMap(ChaoticMap):
    domain = (-1, 1)

    def __init__(self, steps, trans, k=2, x0=None, prec=50, seed=None):
        self.k = int(k)
        super().__init__(steps, trans, x0, prec, seed)

    def f(self, x):
        t0, t1 = D('1'), x
        for _ in range(self.k - 1):
            t0, t1 = t1, D('2') * x * t1 - t0
        return t1

    def df(self, x):
        u0, u1 = D('1'), D('2') * x
        for _ in range(self.k - 2):
            u0, u1 = u1, D('2') * x * u1 - u0
        return D(self.k) * u1

    @property
    def theoretical_lyapunov(self):
        return D(self.k).ln()

    def density(self, x):
        pi = D(str(math.pi))
        return D('1') / (pi * (D('1') - x**2).sqrt())


class GeneralizedBernoulliMap(ChaoticMap):
    domain = (0, 1)

    def __init__(self, steps, trans, m=2, x0=None, prec=50, seed=None):
        self.m = D(str(m))
        super().__init__(steps, trans, x0, prec, seed)

    def f(self, x):
        return (self.m * x) % D('1')

    def df(self, x):
        return self.m

    @property
    def theoretical_lyapunov(self):
        return self.m.ln()

    def density(self, x):
        return D('1')

class KT1Map(ChaoticMap):
    domain = (0, 1)

    def __init__(self, steps, trans, gamma = None, x0=None, prec=50, seed=None):
        self.gamma = D(str(gamma))
        super().__init__(steps, trans, x0, prec, seed)

    def f(self, x):
        g = self.gamma
        if x < g:
            return x / g
        else:
            return (g * x - g**2) / (D('1') - g)

    def df(self, x):
        g = self.gamma
        if x < g:
            return D('1') / g
        else:
            return g / (D('1') - g)

    @property
    def theoretical_lyapunov(self):
        g = self.gamma
        term1 = (D('1') / (D('2') - g)) * (D('1') / g).ln()
        term2 = ((D('1') - g) / (D('2') - g)) * (g / (D('1') - g)).ln()
        return term1 + term2

class KT2Map(ChaoticMap):
    domain = (0, 1)

    def __init__(self, steps, trans, gamma = None, x0=None, prec=50, seed=None):
        self.gamma = D(str(gamma))
        super().__init__(steps, trans, x0, prec, seed)

    def f(self, x):
        g = self.gamma
        if x < D('1') - g:
            return (g * x) / (D('1') - g) + D('1') - g
        else:
            return (x - (D('1') - g)) / g

    def df(self, x):
        g = self.gamma
        if x < D('1') - g:
            return g / (D('1') - g)
        else:
            return D('1') / g

    @property
    def theoretical_lyapunov(self):
        g = self.gamma
        term1 = (D('1') / (D('2') - g)) * (D('1') / g).ln()
        term2 = ((D('1') - g) / (D('2') - g)) * (g / (D('1') - g)).ln()
        return term1 + term2

class Manneville(ChaoticMap):
    domain = (0, 1)

    def __init__(self, steps, trans, epslon = None, x0=None, prec=50, seed=None):
        self.epslon = D(str(epslon))
        super().__init__(steps, trans, x0, prec, seed)

    def f(self, x):
        e = self.epslon
        return ((D('1') + e) * x + (D('1') - e) * x**2) % D('1')

    def df(self, x):
        e = self.epslon
        return (D('1') + e) + D('2') * (D('1') - e) * x

    @property
    def theoretical_lyapunov(self):
        raise NotImplementedError("No closed form available")


    def density(self, x):
        e = self.epslon
        K = (D('1') - e) / ((D('2') - e).ln() - e.ln())
        term1 = D('1') / (e + (D('1') - e) * x)
        term2 = D('1') / (D('1') + (D('1') - e) * x)
        return K * (term1 + term2)

class ConjugateTentMap(ChaoticMap):
    domain = (0, 1)

    def __init__(self, steps, trans, p, x0=None, prec=50, seed=None):
        self.p = D(str(p))
        super().__init__(steps, trans, x0, prec, seed)

    def f(self, x):
        p = self.p
        q = D('1') - p
        if x <= p**2:
            return x / p**2
        else:
            return (D('1') - x.sqrt())**2 / q**2

    def df(self, x):
        p = self.p
        q = D('1') - p
        if x <= p**2:
            return D('1') / p**2
        else:
            return -(D('1') - x.sqrt()) / (q**2 * x.sqrt())

    @property
    def theoretical_lyapunov(self):
        raise NotImplementedError("No closed form available")

    def density(self, x):
        return D('1') / (D('2') * x.sqrt())

class ThalerMap(ChaoticMap):
    domain = (0, 1)

    def __init__(self, steps, trans, z, x0=None, prec=50, seed=None):
        self.z = D(str(z))
        super().__init__(steps, trans, x0, prec, seed)

    def f(self, x):
        z = self.z
        inner = (x / (D('1') + x))**(z - D('2')) - x**(z - D('2'))
        return (x * (D('1') + inner) ** (D('-1') / (z - D('2')))) % D('1')

    def df(self, x):
        z = self.z
        a = z - D('2')                                      # expoente base
        ratio = x / (D('1') + x)                           # x/(1+x)
        g = D('1') + ratio**a - x**a                       # g(x)
        dg = (z - D('2')) * (ratio**(a - D('1')) / (D('1') + x)**2 - x**(a - D('1')))  # g'(x)
        exp = D('-1') / a                                   # -1/(z-2)

        return g**exp - x * g**(exp - D('1')) * dg

    @property
    def theoretical_lyapunov(self):
        raise NotImplementedError("No closed form available")

    def density(self, x):
        z = self.z
        alpha = D('1') / (z - D('1'))
        exp = D('-1') / alpha
        return x**exp + (D('1') + x)**exp
        # não normalizada
