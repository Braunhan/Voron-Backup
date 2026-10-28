#!/usr/bin/env python3
# =============================================================================
# scripts/multi_accel.py — Auswertung für SHAKETUNE_MULTI
# =============================================================================
# Wertet die Rohdaten von TEST_RESONANCES ... CHIPS="lis2dw,adxl345 cartographer"
# aus: beide Sensoren wurden im selben Lauf aufgezeichnet. Nutzt Klippers
# eigene Shaper-Berechnung (klippy/extras/shaper_calibrate.py).
#
# Aufruf (über RUN_SHELL_COMMAND):
#   multi_accel.py shaper <NAME> <SCV>   -> Input-Shaper X/Y, ein Bild pro Achse
#   multi_accel.py belts  <NAME>         -> Riemenvergleich A/B pro Sensor
# =============================================================================
import glob, importlib, os, sys, time
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

KLIPPY = os.path.expanduser('~/klipper/klippy')
sys.path.append(KLIPPY)
shaper_calibrate = importlib.import_module('.shaper_calibrate', 'extras')

OUT_DIR = os.path.expanduser('~/printer_data/config/ShakeTune_results/multi_sensor')
SENSORS = [
    ('lis2dw', 'LIS2DW (EBB36, Druckkopf)', '#7b2cbf'),
    ('cartographer', 'ADXL345 (Cartographer)', '#f77f00'),    # Klipper nutzt nur den Abschnittsnamen
]
SHOW_SHAPERS = ['zv', 'mzv', 'ei', '2hump_ei', '3hump_ei']
MAX_FREQ = 200.


def wait_for_file(path, timeout=60.):
    # Klipper schreibt die Rohdaten in einem Hintergrundprozess
    end = time.time() + timeout
    last = -1
    while time.time() < end:
        if os.path.exists(path):
            size = os.path.getsize(path)
            if size > 0 and size == last:
                return True
            last = size
        time.sleep(1.)
    return os.path.exists(path)


def find_raw(axis_name, chip, name):
    path = '/tmp/raw_data_%s_%s_%s.csv' % (axis_name, chip, name)
    if not wait_for_file(path):
        return None
    return path


def load_calibration(path, normalize):
    data = np.loadtxt(path, comments='#', delimiter=',')
    helper = shaper_calibrate.ShaperCalibrate(printer=None)
    cdata = helper.process_accelerometer_data(path, data)
    if normalize:
        cdata.normalize_to_frequencies()
    return cdata


def find_peaks(freqs, psd, count=3, rel_height=0.15):
    peaks = [i for i in range(1, len(psd) - 1)
             if psd[i] >= psd[i - 1] and psd[i] > psd[i + 1]
             and psd[i] >= rel_height * psd.max()]
    peaks.sort(key=lambda i: psd[i], reverse=True)
    return sorted(freqs[i] for i in peaks[:count])


def stamp():
    return datetime.now().strftime('%Y%m%d_%H%M%S')


# --- Input Shaper ------------------------------------------------------------
def run_shaper(name, scv):
    summary = []
    for axis in ('x', 'y'):
        fig = plt.figure(figsize=(16, 9))
        gs = fig.add_gridspec(2, 1, height_ratios=[3, 2], hspace=0.35)
        ax = fig.add_subplot(gs[0])
        tab_ax = fig.add_subplot(gs[1])
        tab_ax.axis('off')
        rows, row_colors = [], []
        for chip, label, color in SENSORS:
            path = find_raw(axis, chip, name)
            if path is None:
                summary.append('%s-Achse, %s: keine Daten gefunden' % (axis.upper(), label))
                continue
            cdata = load_calibration(path, normalize=True)
            helper = shaper_calibrate.ShaperCalibrate(printer=None)
            best, shapers = helper.find_best_shaper(
                cdata, shapers=SHOW_SHAPERS, scv=scv, max_freq=MAX_FREQ)
            freqs = cdata.freq_bins
            mask = freqs <= MAX_FREQ
            psd = cdata.psd_sum[mask]
            psd_axis = cdata.get_psd(axis)[mask]
            norm = psd.max() if psd.max() > 0 else 1.
            peaks = find_peaks(freqs[mask], psd)
            ax.plot(freqs[mask], psd / norm, color=color, lw=2,
                    label='%s  (Gipfel: %s Hz)' % (label, ', '.join('%.1f' % p for p in peaks)))
            ax.plot(freqs[mask], psd_axis / norm, color=color, lw=1, ls='--', alpha=0.6,
                    label='%s  nur %s-Anteil' % (label, axis.upper()))
            ax.axvline(best.freq, color=color, ls=':', lw=1.5)
            for s in shapers:
                rows.append([label, s.name.upper() + (' (empfohlen)' if s.name == best.name else ''),
                             '%.1f Hz' % s.freq, '%.1f %%' % (s.vibrs * 100.),
                             '%.3f' % s.smoothing, '%.0f' % (round(s.max_accel / 100.) * 100.)])
                row_colors.append(color)
            summary.append('%s-Achse, %s: Gipfel %s Hz -> empfohlen %s @ %.1f Hz '
                           '(Vibration %.1f %%, max. Beschl. %.0f)'
                           % (axis.upper(), label, ', '.join('%.1f' % p for p in peaks),
                              best.name.upper(), best.freq, best.vibrs * 100.,
                              round(best.max_accel / 100.) * 100.))
        ax.set_xlim(0, MAX_FREQ)
        ax.set_ylim(bottom=0)
        ax.set_xlabel('Frequenz (Hz)')
        ax.set_ylabel('Leistungsdichte (normiert je Sensor)')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize='small')
        ax.set_title('Input Shaper %s-Achse – beide Sensoren im selben Lauf   (SCV %.0f mm/s)'
                     % (axis.upper(), scv), fontsize=15, fontweight='bold')
        if rows:
            table = tab_ax.table(cellText=rows,
                                 colLabels=['Sensor', 'Shaper', 'Frequenz', 'Restvibration',
                                            'Glättung', 'max. Beschl. (mm/s²)'],
                                 loc='center', cellLoc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 1.25)
            for (r, c), cell in table.get_celld().items():
                if r == 0:
                    cell.set_text_props(fontweight='bold')
                elif c == 0:
                    cell.set_text_props(color=row_colors[r - 1])
                if r > 0 and '(empfohlen)' in rows[r - 1][1]:
                    cell.set_text_props(fontweight='bold')
        fig.text(0.01, 0.005, 'Gepunktete Linien = empfohlene Shaper-Frequenz je Sensor. '
                 'Kurven sind je Sensor auf ihr Maximum normiert.', fontsize=8, alpha=0.7)
        os.makedirs(OUT_DIR, exist_ok=True)
        out = os.path.join(OUT_DIR, 'shaper_%s_%s.png' % (axis.upper(), stamp()))
        fig.savefig(out, dpi=120, bbox_inches='tight')
        plt.close(fig)
        summary.append('Grafik: %s' % out.replace(os.path.expanduser('~/printer_data/config/'), ''))
    return summary


# --- Riemen ------------------------------------------------------------------
def run_belts(name):
    summary = []
    fig, axes = plt.subplots(len(SENSORS), 1, figsize=(16, 5 * len(SENSORS)), squeeze=False)
    for (chip, label, color), ax in zip(SENSORS, axes[:, 0]):
        curves = {}
        for belt, axis_name in (('A', 'axis=1.000,-1.000,0.000'), ('B', 'axis=1.000,1.000,0.000')):
            path = find_raw(axis_name, chip, name)
            if path is None:
                continue
            cdata = load_calibration(path, normalize=False)
            mask = cdata.freq_bins <= MAX_FREQ
            curves[belt] = (cdata.freq_bins[mask], cdata.psd_sum[mask])
        if len(curves) < 2:
            summary.append('Riemen, %s: Daten unvollständig' % label)
            ax.set_title('%s: Daten unvollständig' % label)
            continue
        fa, pa = curves['A']
        pb = np.interp(fa, *curves['B'])
        similarity = 100. * np.dot(pa, pb) / (np.linalg.norm(pa) * np.linalg.norm(pb))
        peaks_a = find_peaks(fa, pa)
        peaks_b = find_peaks(fa, pb)
        ax.plot(fa, pa, color='#f77f00', lw=2,
                label='Riemen A (unten links, Achse 1,-1)  Gipfel: %s Hz' % ', '.join('%.1f' % p for p in peaks_a))
        ax.plot(fa, pb, color='#7b2cbf', lw=2,
                label='Riemen B (oben rechts, Achse 1,1)  Gipfel: %s Hz' % ', '.join('%.1f' % p for p in peaks_b))
        ax.set_xlim(0, MAX_FREQ)
        ax.set_ylim(bottom=0)
        ax.set_xlabel('Frequenz (Hz)')
        ax.set_ylabel('Leistungsdichte')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize='small')
        ax.set_title('Riemenvergleich – %s   |   Ähnlichkeit %.1f %%' % (label, similarity),
                     fontsize=14, fontweight='bold')
        summary.append('Riemen, %s: Ähnlichkeit %.1f %%, A-Gipfel %s Hz, B-Gipfel %s Hz'
                       % (label, similarity, ', '.join('%.1f' % p for p in peaks_a),
                          ', '.join('%.1f' % p for p in peaks_b)))
    fig.suptitle('Riemenvergleich – beide Sensoren im selben Lauf', fontsize=16, fontweight='bold')
    fig.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, 'belts_%s.png' % stamp())
    fig.savefig(out, dpi=120, bbox_inches='tight')
    plt.close(fig)
    summary.append('Grafik: %s' % out.replace(os.path.expanduser('~/printer_data/config/'), ''))
    return summary


def main():
    if len(sys.argv) < 3 or sys.argv[1] not in ('shaper', 'belts'):
        print('Aufruf: multi_accel.py shaper|belts <NAME> [SCV]')
        sys.exit(1)
    mode, name = sys.argv[1], sys.argv[2]
    if mode == 'shaper':
        scv = float(sys.argv[3]) if len(sys.argv) > 3 else 5.
        lines = run_shaper(name, scv)
    else:
        lines = run_belts(name)
    for line in lines:
        print(line)


if __name__ == '__main__':
    main()
