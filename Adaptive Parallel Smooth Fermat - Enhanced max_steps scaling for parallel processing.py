
# -*- coding: utf-8 -*-
"""
Adaptive Parallel Smooth Fermat - Enhanced max_steps scaling for parallel processing
"""

import math
import random
import time
import sys
from typing import Optional, Tuple, List, Dict
from multiprocessing import Pool, cpu_count
import concurrent.futures

# =========================
# IDENTICAL: Core Math Functions from Original
# =========================

def gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return abs(a)

def is_probable_prime(n: int) -> bool:
    if n < 2:
        return False
    small_primes = [2,3,5,7,11,13,17,19,23,29,31]
    for p in small_primes:
        if n == p:
            return True
        if n % p == 0:
            return n == p
    d = n - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in [2, 325, 9375, 28178, 450775, 9780504, 1795265022]:
        if a % n == 0:
            continue
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        witness = True
        for _ in range(s - 1):
            x = (x * x) % n
            if x == n - 1:
                witness = False
                break
        if witness:
            return False
    return True

def primes_up_to(n: int) -> List[int]:
    if n < 2:
        return []
    sieve = [True]*(n+1)
    sieve[0]=sieve[1]=False
    for i in range(2, int(n**0.5)+1):
        if sieve[i]:
            step = i
            start = i*i
            sieve[start:n+1:step] = [False]*(((n - start)//step) + 1)
    return [i for i, v in enumerate(sieve) if v]

def primes_in_range(lo: int, hi: int) -> List[int]:
    if hi < 2 or hi < lo:
        return []
    ps = primes_up_to(hi)
    return [p for p in ps if p >= max(2, lo)]

def fermat_factor_with_timeout(n: int, time_limit_sec: float = 30.0, max_steps: int = 0) -> Optional[Tuple[int,int,int]]:
    """IDENTICAL to original fermat_factor_with_timeout"""
    if n <= 1:
        return None
    if n % 2 == 0:
        return (2, n//2, 0)
    start = time.time()
    a = math.isqrt(n)
    if a*a < n:
        a += 1
    steps = 0
    while True:
        if max_steps and steps > max_steps:
            return None
        if time.time() - start > time_limit_sec:
            return None
        b2 = a*a - n
        if b2 >= 0:
            b = int(math.isqrt(b2))
            if b*b == b2:
                x = a - b
                y = a + b
                if x*y == n and x>1 and y>1:
                    return (x, y, steps)
        a += 1
        steps += 1

def pollard_rho(n: int, time_limit_sec: float = 10.0) -> Optional[int]:
    """IDENTICAL to original pollard_rho"""
    if n % 2 == 0:
        return 2
    if is_probable_prime(n):
        return n
    start = time.time()
    while time.time() - start < time_limit_sec:
        c = random.randrange(1, n-1)
        f = lambda x: (x*x + c) % n
        x = random.randrange(2, n-1)
        y = x
        d = 1
        while d == 1 and time.time() - start < time_limit_sec:
            x = f(x)
            y = f(f(y))
            d = gcd(abs(x - y), n)
        if 1 < d < n:
            return d
    return None

def modinv(a: int, n: int) -> Tuple[Optional[int], int]:
    """IDENTICAL to original modinv"""
    a = a % n
    if a == 0:
        return (None, n)
    r0, r1 = n, a
    s0, s1 = 1, 0
    t0, t1 = 0, 1
    while r1 != 0:
        q = r0 // r1
        r0, r1 = r1, r0 - q*r1
        s0, s1 = s1, s0 - q*s1
        t0, t1 = t1, t0 - q*t1
    if r0 != 1:
        return (None, r0)
    return (t0 % n, 1)

def ecm_stage1(n: int, B1: int = 10000, curves: int = 50, time_limit_sec: float = 5.0) -> Optional[int]:
    """IDENTICAL to original ecm_stage1"""
    if n % 2 == 0:
        return 2
    if is_probable_prime(n):
        return n

    start = time.time()

    # prime powers up to B1
    smalls = primes_up_to(B1)
    prime_powers = []
    for p in smalls:
        e = 1
        while p**(e+1) <= B1:
            e += 1
        prime_powers.append(p**e)

    def ec_add(P, Q, a, n):
        if P is None:
            return Q
        if Q is None:
            return P
        x1, y1 = P
        x2, y2 = Q
        if x1 == x2 and (y1 + y2) % n == 0:
            return None  # point at infinity
        if x1 == x2 and y1 == y2:
            num = (3 * x1 * x1 + a) % n
            den = (2 * y1) % n
        else:
            num = (y2 - y1) % n
            den = (x2 - x1) % n
        inv, g = modinv(den, n)
        if inv is None:
            if 1 < g < n:
                raise ValueError(g)
            return None
        lam = (num * inv) % n
        x3 = (lam*lam - x1 - x2) % n
        y3 = (lam*(x1 - x3) - y1) % n
        return (x3, y3)

    def ec_mul(k, P, a, n):
        R = None
        Q = P
        while k > 0:
            if k & 1:
                R = ec_add(R, Q, a, n)
            Q = ec_add(Q, Q, a, n)
            k >>= 1
        return R

    while time.time() - start < time_limit_sec and curves > 0:
        x = random.randrange(2, n-1)
        y = random.randrange(2, n-1)
        a = random.randrange(1, n-1)
        b = (pow(y,2,n) - (pow(x,3,n) + a*x)) % n
        disc = (4*pow(a,3,n) + 27*pow(b,2,n)) % n
        g = gcd(disc, n)
        if 1 < g < n:
            return g
        P = (x, y)
        try:
            for k in prime_powers:
                P = ec_mul(k, P, a, n)
                if P is None:
                    break
        except ValueError as e:
            g = int(str(e))
            if 1 < g < n:
                return g
        curves -= 1
    return None

# =========================
# ENHANCED: Smart Max Steps Calculation for Parallel Processing
# =========================

def calculate_enhanced_adaptive_max_steps(N: int, P: int, is_parallel: bool = True, num_workers: int = 1) -> int:
    """
    Enhanced adaptive max_steps calculation with proper parallel processing scaling
    
    Args:
        N: Number to factor
        P: Smooth parameter product  
        is_parallel: Whether running in parallel mode
        num_workers: Number of parallel workers
    
    Returns:
        Enhanced adaptive max_steps value
    """
    digits = len(str(N))
    
    # Base steps scaling by digits (much higher for parallel)
    if is_parallel:
        if digits <= 20:
            base_steps = 5000
        elif digits <= 30:
            base_steps = 10000
        elif digits <= 40:
            base_steps = 20000
        elif digits <= 50:
            base_steps = 50000    # 51 basamak için 50k base
        elif digits <= 60:
            base_steps = 100000
        elif digits <= 70:
            base_steps = 200000
        elif digits <= 80:
            base_steps = 500000
        elif digits <= 90:
            base_steps = 1000000
        else:
            base_steps = 2000000
    else:
        # Single-threaded more conservative
        if digits <= 30:
            base_steps = 1000
        elif digits <= 50:
            base_steps = 5000
        elif digits <= 70:
            base_steps = 20000
        else:
            base_steps = 50000
    
    # Calculate square gap analysis
    _, gap_N = square_proximity(N)
    M = N * P
    _, gap_M = square_proximity(M)
    
    # Gap improvement factor (more conservative scaling)
    if gap_N > 0:
        gap_ratio = gap_M / gap_N
        if gap_ratio > 1e20:
            gap_factor = 0.3  # Excellent improvement
        elif gap_ratio > 1e15:
            gap_factor = 0.5  # Very good improvement  
        elif gap_ratio > 1e12:
            gap_factor = 0.7  # Good improvement
        elif gap_ratio > 1e8:
            gap_factor = 1.0  # Moderate improvement
        else:
            gap_factor = 2.0  # Poor improvement, need more steps
    else:
        gap_factor = 1.0
    
    # P effectiveness factor
    P_digits = len(str(P))
    if P_digits >= 25:  # Large P (like 31 digits in your case)
        p_factor = 0.4   # P helps significantly
    elif P_digits >= 20:
        p_factor = 0.6
    elif P_digits >= 15:
        p_factor = 0.8
    else:
        p_factor = 1.2   # Small P, less help
    
    # Parallel worker scaling
    if is_parallel and num_workers > 1:
        # Each worker gets variation, so base can be slightly lower per worker
        worker_factor = max(0.5, 1.0 - (num_workers - 1) * 0.05)
    else:
        worker_factor = 1.0
    
    # Final calculation
    adaptive_steps = int(base_steps * gap_factor * p_factor * worker_factor)
    
    # Apply bounds with parallel considerations
    if is_parallel:
        min_steps = max(10000, digits * 500)    # Higher minimum for parallel
        max_steps_limit = min(5000000, digits * 50000)
    else:
        min_steps = max(1000, digits * 100)     # Lower for single-thread
        max_steps_limit = min(1000000, digits * 20000)
    
    adaptive_steps = max(min_steps, min(adaptive_steps, max_steps_limit))
    
    return adaptive_steps

def square_proximity(n: int) -> Tuple[int, int]:
    """Return (a, gap) where a=ceil(sqrt(n)), gap=a^2 - n."""
    a = math.isqrt(n)
    if a*a < n:
        a += 1
    gap = a*a - n
    return a, gap

# =========================
# IDENTICAL: Core Smooth Fermat Functions from Original
# =========================

def divide_out_P_from_factors(A: int, B: int, P: int, primesP: List[int]) -> Tuple[int,int]:
    """IDENTICAL to original divide_out_P_from_factors"""
    remP = P
    for p in primesP:
        if remP % p == 0:
            if A % p == 0:
                A //= p
                remP //= p
            elif B % p == 0:
                B //= p
                remP //= p
    return A, B

def factor_with_smooth_fermat(N: int, P: int, P_primes: List[int],
                              time_limit_sec: float = 60.0, max_steps: int = 0,
                              rho_time: float = 10.0, ecm_time: float = 10.0,
                              ecm_B1: int = 20000, ecm_curves: int = 60) -> Optional[Tuple[List[int], dict]]:
    """MODIFIED: Now uses enhanced adaptive max_steps if not specified"""
    if N <= 1:
        return None
    
    # Use enhanced adaptive max_steps if not specified
    if max_steps <= 0:
        max_steps = calculate_enhanced_adaptive_max_steps(N, P, is_parallel=False)
    
    M = N * P
    t0 = time.time()
    res = fermat_factor_with_timeout(M, time_limit_sec=time_limit_sec, max_steps=max_steps)
    t1 = time.time()
    stats = {"method": "enhanced_adaptive_smooth_fermat", "time": t1 - t0, "ok": False, "max_steps_used": max_steps}
    if res is None:
        return None
    A, B, steps = res
    stats["steps"] = steps

    A2, B2 = divide_out_P_from_factors(A, B, P, P_primes)
    if A2*B2 != N:
        g = gcd(A, N)
        if 1 < g < N:
            A2 = g
            B2 = N // g
        else:
            g = gcd(B, N)
            if 1 < g < N:
                A2 = g
                B2 = N // g
            else:
                return None
    stats["ok"] = True

    # Try to break A2,B2 to primes - IDENTICAL logic
    factors = []
    for x in [A2, B2]:
        if x == 1:
            continue
        if is_probable_prime(x):
            factors.append(x)
            continue
        d = pollard_rho(x, time_limit_sec=rho_time)
        if d is None:
            d = ecm_stage1(x, B1=ecm_B1, curves=ecm_curves, time_limit_sec=ecm_time)
        if d is None or d == x:
            rf = fermat_factor_with_timeout(x, time_limit_sec=min(5.0, time_limit_sec), max_steps=max_steps)
            if rf is None:
                factors.append(x)
            else:
                a, b, _ = rf
                for y in (a, b):
                    if is_probable_prime(y):
                        factors.append(y)
                    else:
                        d2 = pollard_rho(y, time_limit_sec=rho_time/2)
                        if d2 and d2 != y:
                            factors.extend([d2, y//d2])
                        else:
                            factors.append(y)
        else:
            z1, z2 = d, x//d
            for z in (z1, z2):
                if is_probable_prime(z):
                    factors.append(z)
                else:
                    d3 = pollard_rho(z, time_limit_sec=rho_time/2)
                    if d3 and d3 != z:
                        factors.extend([d3, z//d3])
                    else:
                        factors.append(z)

    factors.sort()
    return factors, stats

def factor_prime_list(factors: List[int]) -> List[int]:
    """IDENTICAL to original factor_prime_list"""
    out = []
    for f in factors:
        if f == 1:
            continue
        if is_probable_prime(f):
            out.append(f)
        else:
            d = pollard_rho(f, time_limit_sec=5.0)
            if d and 1 < d < f:
                out.extend([d, f//d])
            else:
                out.append(f)
    return sorted(out)

# =========================
# NEW: Enhanced Parallel Wrapper
# =========================

def smooth_fermat_worker(args) -> Optional[Tuple[List[int], Dict]]:
    """
    Enhanced worker function with proper step scaling per worker
    """
    N, P, P_primes, worker_id, time_limit, base_max_steps, num_workers, rho_time, ecm_time, ecm_B1, ecm_curves = args
    
    # Add slight randomization per worker to avoid identical work
    random.seed(worker_id * 12345 + int(time.time() * 1000) % 10000)
    
    # Each worker gets variation around the base steps
    worker_variation = 0.7 + 0.6 * random.random()  # 0.7x to 1.3x variation
    worker_steps = int(base_max_steps * worker_variation)
    
    # Ensure minimum reasonable steps per worker
    digits = len(str(N))
    min_worker_steps = max(5000, digits * 200)
    worker_steps = max(min_worker_steps, worker_steps)
    
    # Slightly vary other parameters per worker
    worker_rho_time = max(2.0, rho_time + random.uniform(-1.0, 1.0))
    worker_ecm_time = max(2.0, ecm_time + random.uniform(-1.0, 1.0))
    worker_ecm_curves = max(10, int(ecm_curves + random.randint(-10, 10)))
    worker_ecm_B1 = max(1000, int(ecm_B1 + random.randint(-1000, 1000)))
    
    # Run algorithm with worker-specific parameters
    return factor_with_smooth_fermat(
        N, P, P_primes,
        time_limit_sec=time_limit,
        max_steps=worker_steps,
        rho_time=worker_rho_time,
        ecm_time=worker_ecm_time,
        ecm_B1=worker_ecm_B1,
        ecm_curves=worker_ecm_curves
    )

def parallel_enhanced_adaptive_smooth_fermat(N: int, P: int, P_primes: List[int],
                                            time_limit_sec: float = 60.0,
                                            max_steps: int = 0,
                                            max_workers: int = None,
                                            rho_time: float = 10.0,
                                            ecm_time: float = 10.0,
                                            ecm_B1: int = 20000,
                                            ecm_curves: int = 60) -> Optional[Tuple[List[int], Dict]]:
    """
    Run enhanced adaptive smooth Fermat algorithm in parallel
    """
    if max_workers is None:
        max_workers = min(cpu_count(), 8)
    
    max_workers = max(1, min(max_workers, cpu_count()))
    
    # Calculate enhanced adaptive max_steps for parallel processing
    if max_steps <= 0:
        adaptive_steps = calculate_enhanced_adaptive_max_steps(N, P, is_parallel=True, num_workers=max_workers)
    else:
        # If user specified, ensure it's reasonable for parallel work
        digits = len(str(N))
        min_parallel_steps = max(10000, digits * 300)
        adaptive_steps = max(max_steps, min_parallel_steps)
    
    print(f"  Starting enhanced parallel smooth Fermat:")
    print(f"    Workers: {max_workers}")
    print(f"    Enhanced adaptive max steps: {adaptive_steps:,}")
    print(f"    Time limit: {time_limit_sec}s")
    print(f"    Steps per worker: ~{adaptive_steps//2:,} to ~{int(adaptive_steps*1.3):,}")
    
    # Create tasks for workers
    tasks = []
    for worker_id in range(max_workers):
        tasks.append((N, P, P_primes, worker_id, time_limit_sec, adaptive_steps, max_workers,
                     rho_time, ecm_time, ecm_B1, ecm_curves))
    
    start_time = time.time()
    
    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_worker = {
                executor.submit(smooth_fermat_worker, task): i 
                for i, task in enumerate(tasks)
            }
            
            # Wait for first successful result
            for future in concurrent.futures.as_completed(future_to_worker, timeout=time_limit_sec + 5):
                worker_id = future_to_worker[future]
                try:
                    result = future.result()
                    if result is not None:
                        elapsed = time.time() - start_time
                        factors, stats = result
                        stats['worker_id'] = worker_id
                        stats['parallel_time'] = elapsed
                        stats['total_workers'] = max_workers
                        stats['base_max_steps'] = adaptive_steps
                        
                        print(f"    SUCCESS by worker {worker_id} in {elapsed:.6f}s")
                        print(f"    Steps used: {stats.get('steps', 0):,}/{stats.get('max_steps_used', adaptive_steps):,}")
                        
                        # Cancel remaining tasks
                        for f in future_to_worker:
                            f.cancel()
                        
                        return factors, stats
                        
                except Exception as e:
                    print(f"    Worker {worker_id} error: {e}")
                    continue
    
    except concurrent.futures.TimeoutError:
        print(f"    Parallel processing timed out after {time_limit_sec}s")
    except Exception as e:
        print(f"    Parallel processing error: {e}")
        print("    Falling back to single-threaded...")
        
        # Fallback to single-threaded with enhanced adaptive algorithm
        single_steps = calculate_enhanced_adaptive_max_steps(N, P, is_parallel=False)
        return factor_with_smooth_fermat(N, P, P_primes, time_limit_sec, single_steps,
                                       rho_time, ecm_time, ecm_B1, ecm_curves)
    
    return None

# =========================
# Interface Functions
# =========================

def calculate_P_from_input(P_input: str) -> Tuple[int, List[int]]:
    """Calculate P from user input"""
    P_input = P_input.strip()
    
    if '-' in P_input:
        lo, hi = map(int, P_input.split('-', 1))
        primes_P = primes_in_range(lo, hi)
    elif ',' in P_input:
        primes_P = [int(x.strip()) for x in P_input.split(',')]
        for p in primes_P:
            if not is_probable_prime(p):
                raise ValueError(f"{p} is not prime")
    else:
        upper_bound = int(P_input)
        primes_P = primes_up_to(upper_bound)
    
    P = 1
    for p in primes_P:
        P *= p
    
    return P, primes_P

def factor_with_enhanced_parallel_smooth_fermat(N: int, P_input: str,
                                               max_workers: int = None,
                                               time_limit_sec: float = 60.0,
                                               max_steps: int = 0,
                                               rho_time: float = 10.0,
                                               ecm_time: float = 10.0,
                                               ecm_B1: int = 20000,
                                               ecm_curves: int = 60) -> Dict:
    """
    Main factorization function using enhanced adaptive parallel smooth Fermat
    """
    # Parse P
    P, P_primes = calculate_P_from_input(P_input)
    
    result = {
        'N': N,
        'P': P,
        'P_primes': P_primes,
        'P_input': P_input,
        'digits': len(str(N)),
        'P_digits': len(str(P)),
        'success': False,
        'factors': None,
        'method': None,
        'time': 0,
        'steps': None,
        'max_steps_used': 0,
        'workers_used': 0
    }
    
    print(f"\nEnhanced Parallel Smooth Fermat Factorization:")
    print(f"  N = {N} ({len(str(N))} digits)")
    print(f"  P_input = {P_input}")
    print(f"  P = {P} ({len(str(P))} digits)")
    print(f"  P_primes = {P_primes}")
    
    # Calculate square proximities and show enhanced adaptive calculation
    _, gap_N = square_proximity(N)
    M = N * P
    _, gap_M = square_proximity(M)
    gap_ratio = gap_M / gap_N if gap_N > 0 else float('inf')
    
    if max_workers == 1:
        adaptive_steps = calculate_enhanced_adaptive_max_steps(N, P, is_parallel=False)
    else:
        if max_workers is None:
            max_workers = min(cpu_count(), 8)
        adaptive_steps = calculate_enhanced_adaptive_max_steps(N, P, is_parallel=True, num_workers=max_workers)
    
    print(f"  Square gap N: {gap_N:,}")
    print(f"  Square gap M: {gap_M:,}")
    print(f"  Gap ratio: {gap_ratio:.2e}")
    print(f"  Enhanced adaptive max steps: {adaptive_steps:,}")
    
    start_time = time.time()
    
    # Run enhanced adaptive parallel smooth Fermat
    if max_workers == 1:
        print("  Using single-threaded enhanced adaptive algorithm")
        sf_result = factor_with_smooth_fermat(
            N, P, P_primes,
            time_limit_sec=time_limit_sec,
            max_steps=adaptive_steps,
            rho_time=rho_time,
            ecm_time=ecm_time,
            ecm_B1=ecm_B1,
            ecm_curves=ecm_curves
        )
        if sf_result:
            factors, stats = sf_result
            stats['parallel_time'] = stats['time']
            stats['total_workers'] = 1
    else:
        sf_result = parallel_enhanced_adaptive_smooth_fermat(
            N, P, P_primes,
            time_limit_sec=time_limit_sec,
            max_steps=max_steps if max_steps > 0 else adaptive_steps,
            max_workers=max_workers,
            rho_time=rho_time,
            ecm_time=ecm_time,
            ecm_B1=ecm_B1,
            ecm_curves=ecm_curves
        )
    
    if sf_result:
        factors, stats = sf_result
        
        # Apply final factorization like original
        factors_final = factor_prime_list(factors)
        
        result['success'] = True
        result['factors'] = factors_final
        result['method'] = 'Enhanced Parallel Smooth Fermat'
        result['time'] = stats.get('parallel_time', stats['time'])
        result['steps'] = stats.get('steps')
        result['max_steps_used'] = stats.get('max_steps_used', adaptive_steps)
        result['workers_used'] = stats.get('total_workers', 1)
        
        print(f"\n✓ SUCCESS!")
        print(f"  Raw factors: {factors}")
        print(f"  Final factors: {factors_final}")
        print(f"  Time: {result['time']:.6f}s")
        print(f"  Steps used: {result['steps']:,}/{result['max_steps_used']:,}")
        print(f"  Workers: {result['workers_used']}")
        print(f"  Step efficiency: {(result['steps']/result['max_steps_used']*100):.1f}%")
        
        # Verify
        product = 1
        for f in factors_final:
            product *= f
        
        if product == N:
            print(f"  ✓ Verification passed!")
        else:
            print(f"  ✗ Verification failed! Product: {product}")
            result['success'] = False
    
    else:
        result['time'] = time.time() - start_time
        print(f"\n✗ FAILED after {result['time']:.2f}s")
    
    return result

# =========================
# Interactive Interface
# =========================

def interactive_mode():
    """Interactive mode with enhanced adaptive parameters"""
    print("=" * 70)
    print("ENHANCED PARALLEL SMOOTH FERMAT FACTORIZATION")
    print("Intelligent max_steps scaling optimized for parallel processing")
    print("=" * 70)
    
    while True:
        try:
            print("\nEnter the number to factor (N):")
            N_input = input("N = ").strip()
            if not N_input:
                break
            
            N = int(N_input)
            digits = len(str(N))
            
            print(f"\nNumber analysis: {digits} digits")
            
            print(f"\nEnter P specification:")
            print("  Examples:")
            print("    '40' = all primes up to 40")
            print("    '1-40' = all primes from 1 to 40") 
            print("    '2,3,5,7,11' = specific primes")
            
            # Suggest P based on digit count
            if digits <= 30:
                suggested_P = "40"
            elif digits <= 50:
                suggested_P = "80"   # Higher for better performance
            elif digits <= 70:
                suggested_P = "100"
            elif digits <= 90:
                suggested_P = "120"
            else:
                suggested_P = "150"
                
            P_input = input(f"P (suggested: {suggested_P}): ").strip()
            if not P_input:
                P_input = suggested_P
                print(f"Using suggested: {P_input}")
            
            print(f"\nParallel processing:")
            workers_input = input(f"Workers (1-{cpu_count()}, default=8): ").strip()
            if workers_input:
                max_workers = int(workers_input)
            else:
                max_workers = min(8, cpu_count())
            
            # Enhanced adaptive time limits based on digits
            if digits <= 40:
                time_limit = 30.0
            elif digits <= 60:
                time_limit = 60.0
            elif digits <= 80:
                time_limit = 120.0
            else:
                time_limit = 300.0
            
            print(f"\nRunning with enhanced adaptive parameters:")
            print(f"  Enhanced time limit: {time_limit}s")
            print(f"  Max steps: ENHANCED ADAPTIVE (automatically calculated)")
            
            # Run factorization
            result = factor_with_enhanced_parallel_smooth_fermat(
                N, P_input,
                max_workers=max_workers,
                time_limit_sec=time_limit,
                max_steps=0,  # Let it be enhanced adaptive
                rho_time=10.0,
                ecm_time=10.0,
                ecm_B1=100000,
                ecm_curves=200
            )
            
            print(f"\nResult summary:")
            print(f"  Success: {result['success']}")
            if result['success']:
                print(f"  Factors: {result['factors']}")
                print(f"  Time: {result['time']:.6f}s")
                print(f"  Steps: {result['steps']:,}/{result['max_steps_used']:,}")
                print(f"  Efficiency: {(result['steps']/result['max_steps_used']*100):.1f}% steps used")
                print(f"  Workers: {result['workers_used']}")
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")
            continue

# =========================
# Main Entry Point
# =========================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced Adaptive Parallel Smooth Fermat')
    parser.add_argument('-n', '--number', type=str, help='Number to factor')
    parser.add_argument('-p', '--primes', type=str, help='P specification')
    parser.add_argument('-w', '--workers', type=int, default=8, help='Number of parallel workers')
    parser.add_argument('-t', '--timeout', type=float, default=0, help='Timeout in seconds (0=adaptive)')
    parser.add_argument('--steps', type=int, default=0, help='Max Fermat steps (0=enhanced adaptive)')
    parser.add_argument('--rho-time', type=float, default=10.0, help='Pollard Rho time limit')
    parser.add_argument('--ecm-time', type=float, default=10.0, help='ECM time limit')
    parser.add_argument('--ecm-B1', type=int, default=100000, help='ECM B1 parameter')
    parser.add_argument('--ecm-curves', type=int, default=200, help='ECM curves')
    
    args = parser.parse_args()
    
    if args.number and args.primes:
        N = int(args.number)
        digits = len(str(N))
        
        # Enhanced adaptive timeout if not specified
        if args.timeout <= 0:
            if digits <= 40:
                timeout = 30.0
            elif digits <= 60:
                timeout = 60.0
            elif digits <= 80:
                timeout = 120.0
            else:
                timeout = 300.0
        else:
            timeout = args.timeout
        
        result = factor_with_enhanced_parallel_smooth_fermat(
            N, args.primes,
            max_workers=args.workers,
            time_limit_sec=timeout,
            max_steps=args.steps,
            rho_time=args.rho_time,
            ecm_time=args.ecm_time,
            ecm_B1=args.ecm_B1,
            ecm_curves=args.ecm_curves
        )
    else:
        interactive_mode()
