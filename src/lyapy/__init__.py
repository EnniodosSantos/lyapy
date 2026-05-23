from .lyapy import (
    ChaoticMap,
    LogisticMap,
    GeneralizedLogisticMap,
    UlamMap,
    GeneralizedUlamMap,
    GaussMap,
    BernoulliMap,
    TentMap,
    AsymetricMap,
    ChebyshevMap,
    GeneralizedBernoulliMap,
    KT1Map,
    KT2Map,
    Manneville,
    ConjugateTentMap,
    ThalerMap,
)

def available_maps():
    def _all_subclasses(cls):
        result = []
        for sub in cls.__subclasses__():
            result.append(sub)
            result.extend(_all_subclasses(sub))
        return result

    maps = sorted(set(m.__name__ for m in _all_subclasses(ChaoticMap)))
    print("Available maps on Lyapy:")
    for m in maps:
        print(f"  - {m}")
