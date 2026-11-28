#!/usr/bin/env python3
"""KR260 demo server: CIFAR-10 through two FPGA depthwise-separable blocks, live.

    sudo python3 server.py [--port 8080] [--export ../model/export] [--eval-n 10000]

Serves a dashboard (static/index.html) plus a small JSON API:
    GET  /api/status            mode (FPGA or SIM), model levels, accuracy, bit-exact count, last benchmark
    GET  /api/sensors           INA260 board power and AMS temperatures, with a 2-minute history
    GET  /api/classify?level=50&skip=1&n=6
                                classify n test images; every FPGA output is checked against
                                the same block computed on the CPU
    POST /api/bench             latency/speedup/energy sweep (~40 s); result cached on disk

At startup a background thread measures accuracy on the board for every sparsity level
(the real pipeline: ARM stem -> FPGA block -> ARM proj -> FPGA block -> ARM head), and
checks every FPGA call bit-exact against the CPU. It yields to interactive requests.

Needs root on the board (UIO and u-dma-buf). Without the bitstream, libdsc runs in SIM
mode and the page says so in red.
"""
import argparse
import base64
import ctypes
import hashlib
import json
import os
import random
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
C_IN, C_OUT = 32, 64

# ---- libdsc -----------------------------------------------------------------------------
lib = ctypes.CDLL(os.path.join(HERE, 'libdsc.so'))
P = np.ctypeslib.ndpointer
U8 = P(np.uint8, flags='C_CONTIGUOUS')
I8 = P(np.int8, flags='C_CONTIGUOUS')
I32 = P(np.int32, flags='C_CONTIGUOUS')
F32 = P(np.float32, flags='C_CONTIGUOUS')
dbl, cint = ctypes.c_double, ctypes.c_int
lib.dsc_open.argtypes = [cint]
lib.dsc_open.restype = cint
lib.dsc_mode.restype = ctypes.c_char_p
lib.dsc_set_irq.argtypes = [cint]
lib.dsc_set_params.argtypes = [cint, I8, I32, I8, I32]
lib.dsc_set_params.restype = cint
lib.dsc_run.argtypes = [cint, U8, cint, cint, U8, ctypes.POINTER(dbl), ctypes.POINTER(cint)]
lib.dsc_run.restype = cint
lib.dsc_run_repeat.argtypes = [cint, cint, cint, cint, ctypes.POINTER(dbl), ctypes.POINTER(dbl)]
lib.dsc_run_repeat.restype = cint
lib.dsc_cpu.argtypes = [cint, U8, cint, U8]
lib.dsc_cpu.restype = dbl
lib.dsc_arm_stem.argtypes = [U8, F32, F32, F32, F32, F32, F32, U8]
lib.dsc_arm_proj.argtypes = [U8, F32, F32, U8]
lib.dsc_now_us.restype = dbl


def f32(a):
    return np.ascontiguousarray(a, dtype=np.float32)


class Level:
    """One sparsity level: ARM weights + the two blocks' weight sets in DMA slots."""

    def __init__(self, d, info, index):
        self.level = info['level']
        self.gpu_acc = info['gpu_test_acc']
        self.block_sparsity = info['pw_block_sparsity']
        a = np.load(os.path.join(d, 'arm.npz'))
        self.mean, self.std = f32(a['mean']), f32(a['std'])
        self.w1, self.b1 = f32(a['stem1_w']), f32(a['stem1_b'])
        self.w2, self.b2 = f32(a['stem2_w']), f32(a['stem2_b'])
        self.wp, self.bp = f32(a['proj_w']).reshape(32, 64), f32(a['proj_b'])
        self.fc_w, self.fc_b = a['fc_w'].astype(np.float64), a['fc_b'].astype(np.float64)
        self.slot = {'b1': 2 * index, 'b2': 2 * index + 1}
        self.sched_len = {}
        for blk in ('b1', 'b2'):
            bd = os.path.join(d, blk)
            rd = lambda n, t: np.ascontiguousarray(np.fromfile(os.path.join(bd, n), dtype=t))
            n = lib.dsc_set_params(self.slot[blk], rd('dw_wt.bin', np.int8), rd('dw_b.bin', '<i4').astype(np.int32),
                                   rd('pw_wt.bin', np.int8), rd('pw_b.bin', '<i4').astype(np.int32))
            if n != info['sched_len'][blk]:
                raise SystemExit(f'level {self.level} {blk}: sched_len {n} != export {info["sched_len"][blk]}')
            self.sched_len[blk] = n


class Demo:
    def __init__(self, args):
        self.args = args
        self.fpga = lib.dsc_open(0)
        if self.fpga < 0:
            raise SystemExit(lib.dsc_mode().decode())
        self.mode = lib.dsc_mode().decode()
        with open(os.path.join(args.export, 'summary.json')) as f:
            summary = json.load(f)
        self.classes = summary['classes']
        self.levels = [Level(os.path.join(args.export, f's{lv["level"]}'), lv, i)
                       for i, lv in enumerate(summary['levels'])]
        t = np.load(os.path.join(args.export, 'test_images.npz'))
        self.x, self.y = np.ascontiguousarray(t['x']), t['y']
        h = hashlib.sha1(json.dumps(summary, sort_keys=True).encode())
        self.key = f'{h.hexdigest()[:12]}-{"fpga" if self.fpga else "sim"}'
        self.cache_path = os.path.join(args.cache, f'dsc_demo_{self.key}.json')
        self.cache = self.load_cache()

        self.work = threading.Lock()        # one image / one benchmark at a time
        self.stats_lock = threading.Lock()
        self.bitexact = {'checked': 0, 'ok': 0}
        self.eval_state = {'running': False, 'level': None, 'done': 0, 'total': 0}
        self.bench_running = False
        self.bench_stage = ''
        self.sensors = Sensors()

    # -- cache: accuracy per level and the last benchmark survive restarts --
    def load_cache(self):
        try:
            with open(self.cache_path) as f:
                return json.load(f)
        except (OSError, ValueError):
            return {'accuracy': {}, 'bench': None}

    def save_cache(self):
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        tmp = self.cache_path + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(self.cache, f)
        os.replace(tmp, self.cache_path)

    def level(self, lv):
        for L in self.levels:
            if L.level == lv:
                return L
        raise KeyError(lv)

    # -- the pipeline --
    def fpga_block(self, L, blk, a, stride, skip, check):
        ho = 16 // stride
        out = np.empty((C_OUT, ho, ho), np.uint8)
        us, n = dbl(), cint()
        rc = lib.dsc_run(L.slot[blk], a, stride, skip, out, ctypes.byref(us), ctypes.byref(n))
        if rc:
            raise RuntimeError(f'dsc_run failed ({rc})')
        r = {'us': us.value, 'sched_len': n.value}
        if check:
            ref = np.empty_like(out)
            r['cpu_us'] = lib.dsc_cpu(L.slot[blk], a, stride, ref)
            r['bitexact'] = bool(np.array_equal(out, ref))
            with self.stats_lock:
                self.bitexact['checked'] += 1
                self.bitexact['ok'] += r['bitexact']
        return out, r

    def infer(self, L, idx, skip=1, check=True):
        t0 = lib.dsc_now_us()
        a1 = np.empty((C_IN, 16, 16), np.uint8)
        lib.dsc_arm_stem(self.x[idx], L.mean, L.std, L.w1, L.b1, L.w2, L.b2, a1)
        t1 = lib.dsc_now_us()
        o1, r1 = self.fpga_block(L, 'b1', a1, 1, skip, check)
        t2 = lib.dsc_now_us()
        a2 = np.empty((C_IN, 16, 16), np.uint8)
        lib.dsc_arm_proj(o1, L.wp, L.bp, a2)
        t3 = lib.dsc_now_us()
        o2, r2 = self.fpga_block(L, 'b2', a2, 2, skip, check)
        t4 = lib.dsc_now_us()
        feat = o2.reshape(C_OUT, -1).mean(axis=1) / 32.0
        logits = L.fc_w @ feat + L.fc_b
        p = np.exp(logits - logits.max())
        p /= p.sum()
        t5 = lib.dsc_now_us()
        cpu_arm = (t1 - t0) + (t3 - t2) + (t5 - t4)
        return dict(pred=int(p.argmax()), probs=p, b1=r1, b2=r2, arm_us=cpu_arm,
                    total_us=t5 - t0 - (r1.get('cpu_us', 0) + r2.get('cpu_us', 0)))

    def classify(self, lv, skip, n):
        L = self.level(lv)
        out = []
        for _ in range(n):
            idx = random.randrange(len(self.y))
            with self.work:
                r = self.infer(L, idx, skip)
            top = np.argsort(r['probs'])[::-1][:3]
            out.append(dict(index=idx, label=int(self.y[idx]), pred=r['pred'],
                            top3=[[int(k), round(float(r['probs'][k]), 3)] for k in top],
                            img=base64.b64encode(self.x[idx].tobytes()).decode(),
                            b1=r['b1'], b2=r['b2'], arm_us=r['arm_us'], total_us=r['total_us']))
        return dict(level=lv, skip=skip, results=out)

    # -- background accuracy evaluation --
    def eval_loop(self):
        n = min(self.args.eval_n, len(self.y))
        for L in self.levels:
            key = str(L.level)
            have = self.cache['accuracy'].get(key)
            if have and have['n'] >= n:
                continue
            correct = 0
            self.eval_state.update(running=True, level=L.level, done=0, total=n)
            for i in range(n):
                while self.bench_running:
                    time.sleep(0.2)
                with self.work:
                    r = self.infer(L, i, skip=1, check=True)
                correct += r['pred'] == int(self.y[i])
                self.eval_state['done'] = i + 1
                if (i + 1) % 500 == 0 or i + 1 == n:
                    self.cache['accuracy'][key] = {'n': i + 1, 'correct': int(correct), 'final': i + 1 == n}
                    self.save_cache()
                time.sleep(0)       # let request threads in
        self.eval_state.update(running=False, level=None)

    # -- benchmark: latency, speedup, energy --
    def bench(self):
        """Per block, at every sparsity level: median latency with skipping on and off,
        then board power while that block runs back to back. Energy per call is
        (busy - idle) power x time per call. The CPU baseline is the same block as plain
        C on one A53 core, measured the same way. Cycles depend only on the weights, so
        one real activation tile per block (from test image 0) is representative."""
        args = self.args
        res = {'time': time.strftime('%Y-%m-%d %H:%M:%S'), 'mode': self.mode, 'fpga': bool(self.fpga),
               'levels': [], 'idle_w': None, 'cpu': {}}
        try:
            with self.work:
                L0 = self.levels[0]
                a1 = np.empty((C_IN, 16, 16), np.uint8)
                lib.dsc_arm_stem(self.x[0], L0.mean, L0.std, L0.w1, L0.b1, L0.w2, L0.b2, a1)
                o1 = np.empty((C_OUT, 16, 16), np.uint8)
                lib.dsc_cpu(L0.slot['b1'], a1, 1, o1)
                a2 = np.empty((C_IN, 16, 16), np.uint8)
                lib.dsc_arm_proj(o1, L0.wp, L0.bp, a2)
                inp = {'b1': (a1, 1), 'b2': (a2, 2)}
                outs = {'b1': np.empty((C_OUT, 16, 16), np.uint8), 'b2': np.empty((C_OUT, 8, 8), np.uint8)}
                window = args.power_s / 2           # per block

                self.bench_stage = 'idle power'
                time.sleep(1.0)
                res['idle_w'] = self.sensors.mean_power(args.power_s)

                for blk, (a, s) in inp.items():
                    self.bench_stage = f'CPU baseline {blk}'
                    t0, n = time.time(), 0
                    while time.time() - t0 < window:
                        lib.dsc_cpu(L0.slot[blk], a, s, outs[blk])
                        n += 1
                    t1 = time.time()
                    res['cpu'][blk] = {'call_us': (t1 - t0) / n * 1e6,
                                       'busy_w': self.sensors.mean_power_between(t0 + 0.3, t1)}

                for L in self.levels:
                    row = {'level': L.level, 'sched_len': L.sched_len, 'block_sparsity': L.block_sparsity}
                    for blk, (a, s) in inp.items():
                        for skip in (1, 0):
                            self.bench_stage = f'{L.level}% pruned: latency {blk}, skip {"on" if skip else "off"}'
                            lat = []
                            for _ in range(args.reps):
                                us, n = dbl(), cint()
                                lib.dsc_run(L.slot[blk], a, s, skip, outs[blk], ctypes.byref(us), ctypes.byref(n))
                                lat.append(us.value)
                            lat.sort()
                            row[f'{blk}_{"skip" if skip else "dense"}_us'] = lat[len(lat) // 2]
                        if self.fpga:
                            self.bench_stage = f'{L.level}% pruned: power {blk}'
                            us, n = dbl(), cint()
                            lib.dsc_run(L.slot[blk], a, s, 1, outs[blk], ctypes.byref(us), ctypes.byref(n))
                            t0, calls, busy = time.time(), 0, 0.0
                            while time.time() - t0 < window:
                                tot, mn = dbl(), dbl()
                                if lib.dsc_run_repeat(L.slot[blk], s, 1, 500, ctypes.byref(tot), ctypes.byref(mn)):
                                    raise RuntimeError('dsc_run_repeat failed')
                                calls += 500
                                busy += tot.value
                            t1 = time.time()
                            row[f'{blk}_power'] = {'call_us': busy / calls,
                                                   'busy_w': self.sensors.mean_power_between(t0 + 0.3, t1)}
                    res['levels'].append(row)
            self.cache['bench'] = res
            self.save_cache()
        finally:
            self.bench_running = False
            self.bench_stage = ''

    def status(self):
        acc = self.cache['accuracy']
        return dict(
            mode=self.mode, fpga=bool(self.fpga), classes=self.classes,
            levels=[dict(level=L.level, gpu_acc=L.gpu_acc, sched_len=L.sched_len,
                         block_sparsity=L.block_sparsity, board_acc=acc.get(str(L.level)))
                    for L in self.levels],
            eval=self.eval_state, bitexact=self.bitexact, bench=self.cache['bench'],
            bench_running=self.bench_running, bench_stage=self.bench_stage,
            test_images=len(self.y))


class Sensors:
    """INA260 (whole-SOM power) and AMS (PS/PL temperature), sampled in the background."""

    def __init__(self):
        self.power_path = self.temp_paths = None
        for h in sorted(os.listdir('/sys/class/hwmon')) if os.path.isdir('/sys/class/hwmon') else []:
            base = f'/sys/class/hwmon/{h}'
            try:
                name = open(f'{base}/name').read().strip()
            except OSError:
                continue
            if name.startswith('ina260') and os.path.exists(f'{base}/power1_input'):
                self.power_path = f'{base}/power1_input'
        self.iio = None
        for d in sorted(os.listdir('/sys/bus/iio/devices')) if os.path.isdir('/sys/bus/iio/devices') else []:
            base = f'/sys/bus/iio/devices/{d}'
            try:
                if open(f'{base}/name').read().strip() == 'ams':
                    self.iio = base
            except OSError:
                pass
        self.hist = []                       # (time, watts)
        self.lock = threading.Lock()
        threading.Thread(target=self.loop, daemon=True).start()

    def power(self):
        if not self.power_path:
            return None
        try:
            return int(open(self.power_path).read()) / 1e6
        except (OSError, ValueError):
            return None

    def temp(self, chan):
        if not self.iio:
            return None
        try:
            rd = lambda s: float(open(f'{self.iio}/in_{chan}_{s}').read())
            return (rd('raw') + rd('offset')) * rd('scale') / 1000.0
        except (OSError, ValueError):
            return None

    def loop(self):
        while True:
            w = self.power()
            if w is not None:
                with self.lock:
                    self.hist.append((time.time(), w))
                    if len(self.hist) > 2400:          # 2 minutes at 20 Hz
                        del self.hist[:400]
            time.sleep(0.05)

    def mean_power_between(self, t0, t1):
        with self.lock:
            v = [w for t, w in self.hist if t0 <= t <= t1]
        return sum(v) / len(v) if v else None

    def mean_power(self, seconds):
        t0 = time.time()
        time.sleep(seconds)
        return self.mean_power_between(t0, time.time())

    def snapshot(self):
        now = time.time()
        with self.lock:
            h = [(t, w) for t, w in self.hist if t >= now - 120]
        hist, last = [], None
        for t, w in h:                       # thin to 4 Hz for the chart
            if last is None or t - last >= 0.25:
                hist.append([round(t - now, 2), round(w, 3)])
                last = t
        return dict(power_w=self.power(), ps_temp_c=self.temp('temp0_ps_temp'),
                    pl_temp_c=self.temp('temp2_pl_temp'), history=hist,
                    available=self.power_path is not None)


# ---- HTTP -------------------------------------------------------------------------------
def make_handler(demo):
    class H(BaseHTTPRequestHandler):
        def log_message(self, fmt, *a):
            pass

        def send(self, code, body, ctype='application/json'):
            data = body if isinstance(body, bytes) else json.dumps(body, default=float).encode()
            self.send_response(code)
            self.send_header('Content-Type', ctype)
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            u = urlparse(self.path)
            q = parse_qs(u.query)
            try:
                if u.path in ('/', '/index.html'):
                    with open(os.path.join(HERE, 'static', 'index.html'), 'rb') as f:
                        return self.send(200, f.read(), 'text/html; charset=utf-8')
                if u.path == '/api/status':
                    return self.send(200, demo.status())
                if u.path == '/api/sensors':
                    return self.send(200, demo.sensors.snapshot())
                if u.path == '/api/classify':
                    if demo.bench_running:
                        return self.send(503, {'error': 'benchmark running'})
                    lv = int(q.get('level', ['0'])[0])
                    skip = 1 if q.get('skip', ['1'])[0] != '0' else 0
                    n = max(1, min(12, int(q.get('n', ['6'])[0])))
                    return self.send(200, demo.classify(lv, skip, n))
                return self.send(404, {'error': 'not found'})
            except (KeyError, ValueError) as e:
                return self.send(400, {'error': f'bad request: {e}'})
            except RuntimeError as e:
                return self.send(500, {'error': str(e)})

        def do_POST(self):
            if urlparse(self.path).path != '/api/bench':
                return self.send(404, {'error': 'not found'})
            if demo.bench_running:
                return self.send(409, {'error': 'already running'})
            demo.bench_running = True
            threading.Thread(target=demo.bench, daemon=True).start()
            return self.send(202, {'started': True})
    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=8080)
    ap.add_argument('--bind', default='0.0.0.0')
    ap.add_argument('--export', default=os.path.join(HERE, '..', 'model', 'export'))
    ap.add_argument('--cache', default='/var/cache/dsc-demo' if os.geteuid() == 0 else
                    os.path.expanduser('~/.cache/dsc-demo'))
    ap.add_argument('--eval-n', type=int, default=10000, help='test images per level for board accuracy')
    ap.add_argument('--reps', type=int, default=200, help='calls per latency point in the benchmark')
    ap.add_argument('--power-s', type=float, default=4.0, help='seconds per power measurement')
    args = ap.parse_args()

    demo = Demo(args)
    print(f'dsc demo: {demo.mode}', flush=True)
    print(f'dsc demo: {len(demo.levels)} levels, {len(demo.y)} test images; cache {demo.cache_path}', flush=True)
    threading.Thread(target=demo.eval_loop, daemon=True).start()
    srv = ThreadingHTTPServer((args.bind, args.port), make_handler(demo))
    print(f'dsc demo: http://{args.bind}:{args.port}/', flush=True)
    srv.serve_forever()


if __name__ == '__main__':
    main()
