"""Strategy registry – full map for binance futures bot."""
from typing import List, Optional, Dict, Any

from .base import BaseStrategy

from .orb import ORBStrategy
from .liquidity_sweep import LiquiditySweepStrategy
from .order_block import OrderBlockStrategy
from .supply_demand import SupplyDemandStrategy
from .fvg import FVGStrategy
from .bos_choch import BOSCHOCHStrategy
from .oi_funding import OIFundingStrategy

from .trend_ma import TrendMAStrategy
from .momentum_rsi import MomentumRSIStrategy
from .pullback_ema import PullbackEMAStrategy
from .range_bounce import RangeBounceStrategy

from .ma_ribbon import MARibbonStrategy
from .stoch_bb import StochBBStrategy
from .pivot_scalp import PivotScalpStrategy
from .ema_rsi_scalp import EMARSIScalpStrategy

from .swing_ma_cross import SwingMACrossStrategy
from .swing_rsi import SwingRSIStrategy
from .fib_pullback import FibPullbackStrategy
from .bb_squeeze import BBSqueezeStrategy

from .premium_discount import PremiumDiscountStrategy

# Optional strategies (chỉ vào map nếu file .py tồn tại)
try:
    from .ema_rsi_macd_day import EMARSIMACDDayStrategy
except Exception:
    EMARSIMACDDayStrategy = None

try:
    from .martin_ema_pullback import MartinEMAPullbackStrategy
except Exception:
    MartinEMAPullbackStrategy = None

try:
    from .vwap_ema9 import VWAPEMA9Strategy
except Exception:
    VWAPEMA9Strategy = None

try:
    from .supertrend import SupertrendStrategy
except Exception:
    SupertrendStrategy = None

try:
    from .lux_ict_confluence import LuxICTConfluenceStrategy
except Exception:
    LuxICTConfluenceStrategy = None

try:
    from .trend_fib_vwap import TrendFibVWAPStrategy
except Exception:
    TrendFibVWAPStrategy = None

try:
    from .ob_fvg_pullback_pd import OBFVGPullbackPDStrategy
except Exception:
    OBFVGPullbackPDStrategy = None

try:
    from .ob_fvg_pullback import OBFVGPullbackStrategy
except Exception:
    try:
        from .ob_fvg_pullback_no_pd import OBFVGPullbackStrategy
    except Exception:
        OBFVGPullbackStrategy = None

try:
    from .liq_sweep_pd_supertrend import LiqSweepPDSupertrendStrategy
except Exception:
    LiqSweepPDSupertrendStrategy = None

try:
    from .swing_confluence import SwingConfluenceStrategy
except Exception:
    SwingConfluenceStrategy = None

try:
    from .scalp_confluence import ScalpConfluenceStrategy
except Exception:
    ScalpConfluenceStrategy = None


STRATEGY_MAP = {
    "ORB": ORBStrategy,
    "LiquiditySweep": LiquiditySweepStrategy,
    "OrderBlock": OrderBlockStrategy,
    "SupplyDemand": SupplyDemandStrategy,
    "FVG": FVGStrategy,
    "BOS_CHOCH": BOSCHOCHStrategy,
    "OI_Funding": OIFundingStrategy,
    "TrendMA": TrendMAStrategy,
    "MomentumRSI": MomentumRSIStrategy,
    "PullbackEMA": PullbackEMAStrategy,
    "RangeBounce": RangeBounceStrategy,
    "MARibbon": MARibbonStrategy,
    "StochBB": StochBBStrategy,
    "PivotScalp": PivotScalpStrategy,
    "EMARSIScalp": EMARSIScalpStrategy,
    "SwingMACross": SwingMACrossStrategy,
    "SwingRSI": SwingRSIStrategy,
    "FibPullback": FibPullbackStrategy,
    "BBSqueeze": BBSqueezeStrategy,
    "PremiumDiscount": PremiumDiscountStrategy,
}

if EMARSIMACDDayStrategy is not None:
    STRATEGY_MAP["EMARSIMACDDay"] = EMARSIMACDDayStrategy
if MartinEMAPullbackStrategy is not None:
    STRATEGY_MAP["MartinEMAPullback"] = MartinEMAPullbackStrategy
if VWAPEMA9Strategy is not None:
    STRATEGY_MAP["VWAPEMA9"] = VWAPEMA9Strategy
if SupertrendStrategy is not None:
    STRATEGY_MAP["Supertrend"] = SupertrendStrategy
if LuxICTConfluenceStrategy is not None:
    STRATEGY_MAP["LuxICTConfluence"] = LuxICTConfluenceStrategy
if TrendFibVWAPStrategy is not None:
    STRATEGY_MAP["TrendFibVWAP"] = TrendFibVWAPStrategy
if OBFVGPullbackPDStrategy is not None:
    STRATEGY_MAP["OBFVGPullbackPD"] = OBFVGPullbackPDStrategy
if OBFVGPullbackStrategy is not None:
    STRATEGY_MAP["OBFVGPullback"] = OBFVGPullbackStrategy
if LiqSweepPDSupertrendStrategy is not None:
    STRATEGY_MAP["LiqSweepPDSupertrend"] = LiqSweepPDSupertrendStrategy
if SwingConfluenceStrategy is not None:
    STRATEGY_MAP["SwingConfluence"] = SwingConfluenceStrategy
if ScalpConfluenceStrategy is not None:
    STRATEGY_MAP["ScalpConfluence"] = ScalpConfluenceStrategy


STRATEGY_ALIASES = {
    "orb": "ORB",
    "liquidity_sweep": "LiquiditySweep",
    "liquiditysweep": "LiquiditySweep",
    "sweep": "LiquiditySweep",
    "order_block": "OrderBlock",
    "orderblock": "OrderBlock",
    "ob": "OrderBlock",
    "supply_demand": "SupplyDemand",
    "supplydemand": "SupplyDemand",
    "sd": "SupplyDemand",
    "fvg": "FVG",
    "bos_choch": "BOS_CHOCH",
    "bos": "BOS_CHOCH",
    "choch": "BOS_CHOCH",
    "oi_funding": "OI_Funding",
    "oi": "OI_Funding",
    "funding": "OI_Funding",
    "trend": "TrendMA",
    "trend_ma": "TrendMA",
    "trendma": "TrendMA",
    "momentum": "MomentumRSI",
    "momentum_rsi": "MomentumRSI",
    "momentumrsi": "MomentumRSI",
    "rsi": "MomentumRSI",
    "pullback": "PullbackEMA",
    "pullback_ema": "PullbackEMA",
    "pullbackema": "PullbackEMA",
    "range": "RangeBounce",
    "range_bounce": "RangeBounce",
    "rangebounce": "RangeBounce",
    "ma_ribbon": "MARibbon",
    "maribbon": "MARibbon",
    "ribbon": "MARibbon",
    "stoch_bb": "StochBB",
    "stochbb": "StochBB",
    "stochastic": "StochBB",
    "pivot": "PivotScalp",
    "pivot_scalp": "PivotScalp",
    "pivotscalp": "PivotScalp",
    "ema_rsi_scalp": "EMARSIScalp",
    "emarsiscalp": "EMARSIScalp",
    "ema_scalp": "EMARSIScalp",
    "swing_ma": "SwingMACross",
    "swing_ma_cross": "SwingMACross",
    "swingmacross": "SwingMACross",
    "swing_rsi": "SwingRSI",
    "swingrsi": "SwingRSI",
    "fib": "FibPullback",
    "fib_pullback": "FibPullback",
    "fibpullback": "FibPullback",
    "bb_squeeze": "BBSqueeze",
    "bbsqueeze": "BBSqueeze",
    "squeeze": "BBSqueeze",
    "premium_discount": "PremiumDiscount",
    "premiumdiscount": "PremiumDiscount",
    "premium": "PremiumDiscount",
    "discount": "PremiumDiscount",
    "ema_rsi_macd_day": "EMARSIMACDDay",
    "emarsimacdday": "EMARSIMACDDay",
    "emarsi_macd": "EMARSIMACDDay",
    "martin": "MartinEMAPullback",
    "martin_ema": "MartinEMAPullback",
    "martin_ema_pullback": "MartinEMAPullback",
    "martinemapullback": "MartinEMAPullback",
    "vwap_ema9": "VWAPEMA9",
    "vwapema9": "VWAPEMA9",
    "vwap": "VWAPEMA9",
    "vwap_ema": "VWAPEMA9",
    "supertrend": "Supertrend",
    "st": "Supertrend",
    "lux_ict": "LuxICTConfluence",
    "luxict": "LuxICTConfluence",
    "lux_ict_confluence": "LuxICTConfluence",
    "trend_fib_vwap": "TrendFibVWAP",
    "trendfibvwap": "TrendFibVWAP",
    "tfv": "TrendFibVWAP",
    "confluence": "TrendFibVWAP",
    "ob_fvg_pullback_pd": "OBFVGPullbackPD",
    "obfvgpullbackpd": "OBFVGPullbackPD",
    "ob_fvg_pullback": "OBFVGPullback",
    "obfvgpullback": "OBFVGPullback",
    "liq_sweep_pd_supertrend": "LiqSweepPDSupertrend",
    "liqsweeppdsupertrend": "LiqSweepPDSupertrend",
    "liq_pd_st": "LiqSweepPDSupertrend",
    "swing_confluence": "SwingConfluence",
    "swingconfluence": "SwingConfluence",
    "swing_conf": "SwingConfluence",
    "scalp_confluence": "ScalpConfluence",
    "scalpconfluence": "ScalpConfluence",
    "scalp_conf": "ScalpConfluence",
}


def normalize_strategy_name(name: str) -> Optional[str]:
    if not name:
        return None
    n = str(name).strip()
    if n in STRATEGY_MAP:
        return n
    return STRATEGY_ALIASES.get(n.lower())


def get_all_strategies(
    params: dict = None,
    enabled: Optional[List[str]] = None,
    strategy_params: Optional[Dict[str, dict]] = None,
) -> List[BaseStrategy]:
    params = params or {}
    strategy_params = strategy_params or {}

    if enabled is None:
        names = list(STRATEGY_MAP.keys())
    else:
        names = []
        for raw in enabled:
            canon = normalize_strategy_name(raw)
            if canon and canon not in names:
                names.append(canon)

    strategies = []
    for name in names:
        cls = STRATEGY_MAP.get(name)
        if not cls:
            continue
        p = dict(params)
        p.update(strategy_params.get(name, {}))
        try:
            strategies.append(cls(p))
        except Exception:
            continue
    return strategies


def list_strategy_names() -> List[str]:
    return list(STRATEGY_MAP.keys())
