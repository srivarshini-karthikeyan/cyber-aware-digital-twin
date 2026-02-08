# 🎯 Attack Scenarios Documentation

## Overview

This document describes the cyberattack scenarios implemented in the system.

---

## 🔴 Attack Scenario 1: Sensor Spoofing

### Description
**Attack Type**: Active sensor manipulation  
**Stealth Level**: Medium  
**Impact**: High

### Attack Mechanism
- Attacker compromises level sensor (LIT101)
- Sensor reports **false low level** (e.g., 30% of actual)
- Controller believes level is low
- Inlet valve stays **open** (controller thinks it needs to fill)
- Actual level continues to **rise**
- **Overflow risk** increases

### Example Timeline
```
t=0-10s:  Normal operation (level = 500mm, valve open)
t=10s:    ATTACK STARTS - Sensor compromised
t=10-70s: Sensor reports 150mm (actual = 500-600mm)
          Controller keeps valve open
          Actual level rises to 800mm (overflow threshold)
t=70s:    ATTACK ENDS
t=70-80s: Sensor reports correctly
          Controller finally detects high level
          Valve closes (too late - overflow occurred)
```

### Digital Twin Detection
- **Expected State**: Level should be ~600mm (valve open, pump off)
- **Observed State**: Sensor reports 150mm
- **Divergence**: **High** (>0.5)
- **Safety State**: **UNSAFE** (actual level > 800mm)

### GenAI Detection
- Reconstruction error: **High** (sensor reading doesn't match normal pattern)
- Confidence score: **0.85** (high anomaly)
- Detection delay: **~15 seconds** (GenAI needs sequence window)

### Identified Gaps
1. **Single Sensor Dependency**: No redundant sensor to catch spoofing
2. **Blind Controller Trust**: Controller trusts sensor without validation
3. **No Digital Twin Validation**: Expected vs Observed not compared
4. **Missing Sanity Check**: Impossible level decrease not caught

### Mitigation Recommendations
1. **Priority 1**: Add redundant sensors with voting logic
2. **Priority 1**: Integrate digital twin as validation layer
3. **Priority 2**: Add rate-of-change monitoring
4. **Priority 2**: Implement fail-safe valve closure on divergence

---

## 🟠 Attack Scenario 2: Slow Manipulation (Stealthy)

### Description
**Attack Type**: Gradual sensor drift  
**Stealth Level**: High  
**Impact**: Medium-High

### Attack Mechanism
- Attacker gradually manipulates sensor readings
- Drift is **slow** (evades immediate detection)
- No sudden jumps (doesn't trigger rate checks)
- Over time, sensor reading drifts away from actual
- Controller makes decisions based on **false information**

### Example Timeline
```
t=0-20s:   Normal operation (level = 500mm)
t=20s:     ATTACK STARTS - Gradual drift begins
t=20-140s: Sensor drifts: 500mm → 400mm → 300mm → 200mm
          (Actual level stays ~500mm)
          Drift rate: ~1% per second
          Controller thinks level is dropping
          Keeps valve open longer than needed
t=140s:    ATTACK ENDS
t=140-160s: Sensor returns to normal
```

### Digital Twin Detection
- **Expected State**: Level should be ~500mm (steady state)
- **Observed State**: Sensor reports 200mm (drifted)
- **Divergence**: **Medium** (grows over time)
- **Safety State**: **WARNING** (approaching unsafe)

### GenAI Detection
- Reconstruction error: **Medium** (gradual drift is harder to detect)
- Confidence score: **0.45** (moderate anomaly)
- Detection delay: **~60 seconds** (needs longer sequence to detect trend)

### Identified Gaps
1. **No Rate Validation**: Slow drift not caught by rate checks
2. **Absolute Threshold Only**: No trend-based warnings
3. **No Cross-Sensor Check**: Inconsistencies not validated
4. **Silent Failure**: Degradation without alarm

### Mitigation Recommendations
1. **Priority 1**: Add trend analysis and predictive thresholds
2. **Priority 2**: Implement cross-sensor consistency checks
3. **Priority 2**: Add degradation rate monitoring
4. **Priority 3**: Enhance GenAI with longer sequence windows

---

## 🟡 Attack Scenario 3: Frozen Sensor

### Description
**Attack Type**: Sensor stuck at constant value  
**Stealth Level**: Low  
**Impact**: Medium

### Attack Mechanism
- Attacker freezes sensor at one value
- Sensor reports **same value** regardless of actual level
- Controller thinks level is **constant**
- Actual level continues to change
- Controller makes wrong decisions

### Example Timeline
```
t=0-10s:   Normal operation (level = 500mm, rising)
t=10s:     ATTACK STARTS - Sensor frozen at 500mm
t=10-70s:  Actual level: 500mm → 650mm (valve open)
          Sensor reports: 500mm (frozen)
          Controller thinks level is stable
          Valve stays open (overflow risk)
t=70s:     ATTACK ENDS
t=70-80s:  Sensor reports correctly (650mm)
          Controller detects high level
          Valve closes
```

### Digital Twin Detection
- **Expected State**: Level should be ~650mm (valve open)
- **Observed State**: Sensor reports 500mm (frozen)
- **Divergence**: **High** (grows over time)
- **Safety State**: **UNSAFE** (actual level > 800mm)

### GenAI Detection
- Reconstruction error: **High** (frozen value doesn't match expected pattern)
- Confidence score: **0.75** (high anomaly)
- Detection delay: **~20 seconds** (needs sequence to detect lack of change)

### Identified Gaps
1. **No Rate Validation**: Zero rate-of-change not checked
2. **Missing Sanity Check**: Impossible constant level not caught
3. **No Digital Twin Validation**: Expected vs Observed divergence ignored
4. **Blind Controller Trust**: Controller accepts frozen value

### Mitigation Recommendations
1. **Priority 1**: Add rate-of-change monitoring (detect zero change)
2. **Priority 1**: Implement digital twin validation
3. **Priority 2**: Add sensor health monitoring
4. **Priority 2**: Implement timeout-based sensor validation

---

## 🟢 Attack Scenario 4: Delayed Response

### Description
**Attack Type**: Sensor reports old values  
**Stealth Level**: Medium  
**Impact**: Medium

### Attack Mechanism
- Attacker delays sensor response
- Sensor reports values from **5 seconds ago**
- Controller reacts to **outdated information**
- Control decisions are based on **stale data**
- System becomes unstable

### Example Timeline
```
t=0-20s:   Normal operation
t=20s:     ATTACK STARTS - Sensor delay introduced
t=20-80s:  Actual level: 500mm → 700mm (rising)
          Sensor reports: 500mm → 650mm (5s delay)
          Controller reacts to old values
          Delayed valve closure
          Overshoot risk
t=80s:     ATTACK ENDS
t=80-100s: Sensor reports correctly
          Controller catches up
```

### Digital Twin Detection
- **Expected State**: Level should be ~700mm (current)
- **Observed State**: Sensor reports 650mm (5s old)
- **Divergence**: **Medium** (depends on rate of change)
- **Safety State**: **WARNING** (approaching unsafe)

### GenAI Detection
- Reconstruction error: **Medium** (delay creates temporal mismatch)
- Confidence score: **0.55** (moderate anomaly)
- Detection delay: **~30 seconds** (needs sequence to detect delay pattern)

### Identified Gaps
1. **No Temporal Validation**: Delayed responses not checked
2. **No Cross-Sensor Check**: Inconsistencies with other sensors not validated
3. **Missing Sanity Check**: Impossible temporal patterns not caught
4. **Delayed Response Acceptance**: System accepts stale data

### Mitigation Recommendations
1. **Priority 1**: Add temporal consistency checks
2. **Priority 2**: Implement timestamp validation
3. **Priority 2**: Add cross-sensor temporal alignment
4. **Priority 3**: Enhance GenAI with temporal pattern detection

---

## 📊 Attack Comparison Table

| Attack Type | Stealth | Impact | Detection Difficulty | Common Gaps |
|------------|---------|--------|---------------------|-------------|
| Sensor Spoofing | Medium | High | Easy | Single sensor, blind trust |
| Slow Manipulation | High | Medium-High | Hard | No trend analysis, silent failure |
| Frozen Sensor | Low | Medium | Medium | No rate check, missing sanity |
| Delayed Response | Medium | Medium | Medium | No temporal validation |

---

## 🔍 Detection Metrics

### Average Detection Delays
- Sensor Spoofing: **~15 seconds**
- Slow Manipulation: **~60 seconds**
- Frozen Sensor: **~20 seconds**
- Delayed Response: **~30 seconds**

### Unsafe State Occurrence
- Sensor Spoofing: **Yes** (overflow risk)
- Slow Manipulation: **Sometimes** (depends on duration)
- Frozen Sensor: **Yes** (overflow risk)
- Delayed Response: **Rare** (overshoot risk)

---

## 🛡️ Mitigation Effectiveness

### Before Mitigation
- Average detection delay: **~31 seconds**
- Unsafe state occurrence: **75% of attacks**
- Gaps identified: **4-6 per attack**

### After Mitigation (Simulated)
- Average detection delay: **~8 seconds** (74% reduction)
- Unsafe state occurrence: **20% of attacks** (73% reduction)
- Gaps identified: **1-2 per attack** (residual gaps)

---

## 🎯 Key Insights

1. **Single Sensor Dependency** is the most common gap
2. **Slow Manipulation** is hardest to detect (stealthy)
3. **Digital Twin Validation** would catch most attacks early
4. **Rate-of-Change Monitoring** is critical for frozen sensors
5. **Trend Analysis** is needed for slow manipulation

---

**Status**: ✅ Attack Scenarios Documented | 🎯 Ready for Demonstration
