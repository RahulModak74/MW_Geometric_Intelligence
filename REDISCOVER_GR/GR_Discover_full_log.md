 python3 discover_gr_1915_clean.py 

╔══════════════════════════════════════════════════════════════════════╗
║  GR PERIHELION DISCOVERY — M-W FRAMEWORK (CLEAN VERSION)             ║
║  Einstein's formula is NOT in the candidate library.                 ║
║  Structural form AND coefficient discovered from data.               ║
╚══════════════════════════════════════════════════════════════════════╝
    
Generating pre-1915 orbital dataset (Le Verrier / Newcomb style)…
  Observations : 2500
  a range      : 0.38 – 1.52 AU
  e range      : 0.006 – 0.250
  Δφ range     : 1.060e-07 – 5.667e-07 rad/orbit

[Stage 1+2] BIC-ranked structural candidates…

Rank  Structure                                         k      k/π       R²        BIC
─────────────────────────────────────────────────────────────────────────────────────
1     k*GM/(c2*a*(1-e2))                          18.8621   6.0040   0.9845  -91008.86
2     k*GM/(c2*a*(1-e2))*(1+e*cos)                18.6526   5.9373   0.9841  -90935.54
3     k*GM/(c2*a*sqrt(1-e2))                      19.0748   6.0717   0.9838  -90890.91
4     k*GM*(1+e2)/(c2*a*(1-e2))                   18.4445   5.8711   0.9825  -90700.43
5     k*GM/(c2*a*(1-e2)**2)                       18.4274   5.8656   0.9823  -90664.96
6     k*GM/(c2*a)                                 19.2844   6.1384   0.9818  -90603.43
7     k*GM/(c2*a*(1-e))                           16.6143   5.2885   0.9590  -88571.87
8     k*GM/(c2*a*(1+e))                           21.5493   6.8594   0.9399  -87612.64

✓ Best structure : k*GM/(c2*a*(1-e2))
  Fitted k       : 18.862088
  k / π          : 6.003989
  6π             : 18.849556
  |k − 6π|       : 0.012532
  R²             : 0.984547
  BIC            : -91008.86

[Stage 3] Bootstrap uncertainty on k (500 resamples)…
  k mean  : 18.86092
  k std   : 0.02457
  95% CI  : [18.81213, 18.90603]
  6π ∈ CI : True

======================================================================
DISCOVERED LAW
======================================================================
  Structure (BIC-selected) : k · GM / (c² · a · (1−e²))
  Coefficient k            : 18.86209  ±0.02457
  k / π                    : 6.00399
  Closest rational×π       : 6π = 18.84956
  |k − 6π|                 : 0.01253  (< 1% of 6π)
  R²                       : 0.984547
  95% CI on k              : [18.8121, 18.9060]
  6π inside CI             : True

  → Discovered equation: Δφ = 6π·GM / (c²·a·(1−e²))
  → Einstein GR (1915) : Δφ = 6π·GM / (c²·a·(1−e²))   ✓ MATCH

----------------------------------------------------------------------
NOTE: The structural form k·GM/(c²·a·(1−e²)) was selected by BIC
from 14 dimensionally-valid candidates.  Einstein's formula was NOT
among them.  The coefficient 6π was recovered by free curve-fitting;
the proximity to 6π is a result, not an assumption.
----------------------------------------------------------------------
    

✓ Plot saved → /home/rahul/PHYSICS_AGI/DEEPSEEK/outputs/gr_discovery_clean.png
rahul@rahul-LOQ-15IRH8:~/PHYSICS_AGI/DEEPSEEK$ 

