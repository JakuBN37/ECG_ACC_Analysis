import wfdb
import json
import os
import numpy as np
from scipy.signal import find_peaks, butter, filtfilt
import matplotlib.pyplot as plt
import argparse
from matplotlib.table import Table
from matplotlib.animation import FuncAnimation
import time


# Funkcje Ładowania Danych

def load_acc(file_path):
    record = wfdb.rdrecord(file_path)
    signal = record.p_signal  
    fs = record.fs  
    return signal, fs


def load_ecg(file_path):
    record = wfdb.rdrecord(file_path)
    try:
        annotation = wfdb.rdann(file_path, 'atr') 
    except FileNotFoundError:
        annotation = None
    signal = record.p_signal[:, 0] 
    fs = record.fs 
    return signal, fs, annotation


def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    if "timestamp" in data and "sensor_x" in data:
        timestamps = np.array(data["timestamp"])
        sensor_x = np.array(data["sensor_x"])
        sensor_y = np.array(data["sensor_y"])
        sensor_z = np.array(data["sensor_z"])
        signal = np.column_stack((sensor_x, sensor_y, sensor_z))
        fs = 1 / np.mean(np.diff(timestamps))  
        return signal, fs
    elif "ecg" in data:
        signal = np.array(data["ecg"])
        fs = data.get("fs", 125)  
        return signal, fs, None
    else:
        raise ValueError("Nieobsługiwana struktura pliku JSON.")


def load_data(file_path, data_type=None):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Plik {file_path} nie istnieje.")

    if file_path.endswith(".hea"):
        if data_type == "ecg":
            return load_ecg(file_path.removesuffix(".hea"))
        elif data_type == "acc":
            return load_acc(file_path.removesuffix(".hea"))
        else:
            raise ValueError("Dla plików .hea należy podać typ danych (ecg/acc).")
    elif file_path.endswith(".json"):
        return load_json(file_path)
    else:
        raise ValueError(f"Nieobsługiwany format pliku: {file_path}.")


# Analiza EKG (w tym QRS)

def bandpass_filter(signal, fs, lowcut=0.5, highcut=50.0, order=1):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered_signal = filtfilt(b, a, signal)
    return (filtered_signal - np.mean(filtered_signal)) / np.std(filtered_signal)


def detect_qrs(signal, fs):
    dynamic_height = np.percentile(signal, 90)
    dynamic_distance = int(0.5 * fs)
    r_peaks, properties = find_peaks(signal, distance=dynamic_distance, height=dynamic_height)
    return r_peaks


def analyze_ekg_parameters(r_peaks, signal, fs):
    norms = {
        "PQ": (0.12, 0.20),  # w sekundach
        "QRS": (0.06, 0.11),  # w sekundach
        "QTc": (0.36, 0.44),  # w sekundach
        "P Amplitude": (0, 2.5),  # w mV
        "R Amplitude": (0, 20)  # w mV
    }
    results = []

    for i in range(1, len(r_peaks) - 1):
        r_peak = r_peaks[i]
        q_start = max(0, r_peak - int(0.15 * fs))
        s_end = min(len(signal), r_peak + int(0.15 * fs))

        r_amplitude = signal[r_peak]
        p_amplitude = np.max(signal[q_start:r_peak]) if q_start < r_peak else 0
        qrs_duration = (s_end - q_start) / fs
        pq_duration = (r_peak - q_start) / fs if q_start < r_peak else None
        t_end = min(len(signal), r_peak + int(0.4 * fs))
        qt_duration = (t_end - q_start) / fs

        
        results.append({
            "PQ Duration": {"value": pq_duration, "normal": norms["PQ"][0] <= pq_duration <= norms["PQ"][1] if pq_duration else False},
            "QRS Duration": {"value": qrs_duration, "normal": norms["QRS"][0] <= qrs_duration <= norms["QRS"][1]},
            "QTc Duration": {"value": qt_duration, "normal": norms["QTc"][0] <= qt_duration <= norms["QTc"][1]},
            "P Amplitude": {"value": p_amplitude, "normal": norms["P Amplitude"][0] <= p_amplitude <= norms["P Amplitude"][1]},
            "R Amplitude": {"value": r_amplitude, "normal": norms["R Amplitude"][0] <= r_amplitude <= norms["R Amplitude"][1]},
        })

    return results

def display_ekg_analysis_table(ekg_analysis):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_axis_off()

    table = Table(ax, bbox=[0, 0, 1, 1])

    
    columns = ["Zespół QRS", "PQ Duration (s)", "QRS Duration (s)", "QTc Duration (s)", "P Amplitude (mV)", "R Amplitude (mV)", "Status"]

    
    for col_index, col_name in enumerate(columns):
        table.add_cell(0, col_index, width=0.15, height=0.1, text=col_name, loc='center', facecolor='lightgrey')

    
    for row_index, analysis in enumerate(ekg_analysis):
        status = "Normalne" if all(param["normal"] for param in analysis.values()) else "Nieprawidłowe"
        
        row_data = [
            f"QRS {row_index+1}",
            f"{analysis['PQ Duration']['value']:.3f}" if analysis['PQ Duration']['value'] else "-",
            f"{analysis['QRS Duration']['value']:.3f}",
            f"{analysis['QTc Duration']['value']:.3f}",
            f"{analysis['P Amplitude']['value']:.2f}",
            f"{analysis['R Amplitude']['value']:.2f}",
            status
        ]

        for col_index, value in enumerate(row_data):
            table.add_cell(row_index + 1, col_index, width=0.15, height=0.1, text=value, loc='center')

    ax.add_table(table)
    plt.title("Wyniki Analizy Parametrów EKG", fontsize=14, fontweight="bold")
    plt.show()



def detect_falls(sensor_signal, fs, peak_threshold=8.0, low_amplitude=2.0, flat_duration=0.5, min_distance=1.0):

    magnitude = np.linalg.norm(sensor_signal, axis=1)

    peaks, _ = find_peaks(magnitude, height=peak_threshold, distance=int(min_distance * fs))

    confirmed_falls = []

    for peak in peaks:
        
        post_peak_start = peak + 1
        post_peak_end = peak + int(flat_duration * fs)

        if post_peak_end >= len(magnitude):
            continue  

        post_peak_values = magnitude[post_peak_start:post_peak_end]

        
        if np.all(post_peak_values < low_amplitude) and np.std(post_peak_values) < 0.5:
            confirmed_falls.append(peak)


    return confirmed_falls

def detect_steps(sensor_signal, fs, threshold=1.0, min_distance=0.5):
    magnitude = np.linalg.norm(sensor_signal, axis=1)
    step_peaks, _ = find_peaks(magnitude, height=threshold, distance=int(min_distance * fs))
    return step_peaks


def descriptive_statistics(sensor_signal):
    magnitude = np.linalg.norm(sensor_signal, axis=1)
    stats = {
        "Mean Magnitude": np.mean(magnitude),
        "Max Magnitude": np.max(magnitude),
        "Standard Deviation": np.std(magnitude),
        "Min Magnitude": np.min(magnitude)
    }
    return stats


def analyze_accelerometer(sensor_signal, fs):
    falls = detect_falls(sensor_signal, fs)
    steps = detect_steps(sensor_signal, fs)
    stats = descriptive_statistics(sensor_signal)
    return stats, falls, steps

def simulate_online_analysis(ecg_signal, ecg_fs, sensor_signal, sensor_fs, segment_duration=10):
    num_segments_ecg = len(ecg_signal) // (segment_duration * ecg_fs)
    num_segments_acc = len(sensor_signal) // (segment_duration * sensor_fs)
    total_segments = min(num_segments_ecg, num_segments_acc)

   
    fig, axs = plt.subplots(5, 1, figsize=(12, 15))  
    fig_table = plt.figure(figsize=(8, 10))           
    ax_table = fig_table.add_subplot(111)            

    plt.subplots_adjust(hspace=0.7)  

    def update(frame):
        
        start_ecg = frame * segment_duration * ecg_fs
        end_ecg = start_ecg + segment_duration * ecg_fs

        start_acc = frame * segment_duration * sensor_fs
        end_acc = start_acc + segment_duration * sensor_fs

       
        current_ecg = ecg_signal[start_ecg:end_ecg]
        filtered_ecg = bandpass_filter(current_ecg, ecg_fs)
        r_peaks = detect_qrs(filtered_ecg, ecg_fs)
        ekg_analysis = analyze_ekg_parameters(r_peaks, filtered_ecg, ecg_fs)

        
        current_acc = sensor_signal[start_acc:end_acc]
        magnitude = np.linalg.norm(current_acc, axis=1)
        acc_falls = detect_falls(current_acc, sensor_fs)
        acc_steps = detect_steps(current_acc, sensor_fs)
        time_acc = np.arange(len(current_acc)) / sensor_fs

       
        axs[0].clear()
        axs[0].plot(np.arange(len(current_ecg)) / ecg_fs, filtered_ecg, color='orange', label="EKG")
        axs[0].scatter(np.array(r_peaks) / ecg_fs, filtered_ecg[r_peaks], color='red', label="QRS Peaks")
        axs[0].set_title(f'Segment {frame + 1}: Analiza EKG')
        axs[0].set_xlabel('Czas [s]')
        axs[0].set_ylabel('Amplituda')
        axs[0].legend()
        axs[0].grid()

        
        axes_labels = ['X', 'Y', 'Z']
        colors = ['blue', 'green', 'purple']
        for i in range(3):
            axs[i + 1].clear()
            axs[i + 1].plot(time_acc, current_acc[:, i], color=colors[i], label=f'Oś {axes_labels[i]}')
            axs[i + 1].set_title(f'Segment {frame + 1}: Akcelerometr (Oś {axes_labels[i]})')
            axs[i + 1].set_xlabel('Czas [s]')
            axs[i + 1].set_ylabel('Przyspieszenie [m/s²]')
            axs[i + 1].legend()
            axs[i + 1].grid()

        
        axs[4].clear()
        axs[4].plot(time_acc, magnitude, color='magenta', label='Magnitude')
        axs[4].scatter(np.array(acc_steps) / sensor_fs, magnitude[acc_steps], color='orange', label="Kroki")
        axs[4].scatter(np.array(acc_falls) / sensor_fs, magnitude[acc_falls], color='red', label="Upadki")
        axs[4].set_title(f'Segment {frame + 1}: Magnitude z Krokami i Upadkami')
        axs[4].set_xlabel('Czas [s]')
        axs[4].set_ylabel('Przyspieszenie [m/s²]')
        axs[4].legend()
        axs[4].grid()

        
        acc_stats = descriptive_statistics(current_acc)
        
        
        ax_table.clear()
        ax_table.set_axis_off()

        
        display_ekg_analysis_table(ax_table, ekg_analysis, acc_stats)
        fig_table.canvas.draw()

    ani = FuncAnimation(fig, update, frames=total_segments, interval=segment_duration * 1000, repeat=False)
    plt.show()

def display_ekg_analysis_table(ax, ekg_analysis, acc_stats):
    ax.clear()
    ax.set_axis_off()

    
    columns_ekg = ["QRS", "PQ Duration", "QRS Duration", "QTc Duration", "P Amplitude", "R Amplitude"]
    table_data_ekg = []

   
    for i, analysis in enumerate(ekg_analysis):
        row = [
            f"QRS {i+1}",
            f"{analysis['PQ Duration']['value']:.3f} s - {'Normalne' if analysis['PQ Duration']['normal'] else 'Nieprawidłowe'}",
            f"{analysis['QRS Duration']['value']:.3f} s - {'Normalne' if analysis['QRS Duration']['normal'] else 'Nieprawidłowe'}",
            f"{analysis['QTc Duration']['value']:.3f} s - {'Normalne' if analysis['QTc Duration']['normal'] else 'Nieprawidłowe'}",
            f"{analysis['P Amplitude']['value']:.2f} mV - {'Normalne' if analysis['P Amplitude']['normal'] else 'Nieprawidłowe'}",
            f"{analysis['R Amplitude']['value']:.2f} mV - {'Normalne' if analysis['R Amplitude']['normal'] else 'Nieprawidłowe'}",
        ]
        table_data_ekg.append(row)

    
    table_ekg = ax.table(cellText=table_data_ekg, colLabels=columns_ekg, loc='center', cellLoc='center', colWidths=[0.1, 0.15, 0.15, 0.15, 0.15, 0.15])
    table_ekg.auto_set_font_size(False)
    table_ekg.set_fontsize(10)
    table_ekg.scale(1.2, 1.2)

    
    columns_acc = ["Mean Magnitude", "Max Magnitude", "Standard Deviation", "Min Magnitude"]
    table_data_acc = [
        [f"{acc_stats['Mean Magnitude']:.2f}", f"{acc_stats['Max Magnitude']:.2f}", f"{acc_stats['Standard Deviation']:.2f}", f"{acc_stats['Min Magnitude']:.2f}"]
    ]

    
    table_acc = ax.table(cellText=table_data_acc, colLabels=columns_acc, loc='bottom', cellLoc='center', bbox=[0, -0.1, 1, 0.3])
    table_acc.auto_set_font_size(False)
    table_acc.set_fontsize(10)
    table_acc.scale(1.2, 1.2)

def update_ekg_table(ax, ekg_analysis):
    ax.clear()
    ax.set_axis_off()

    
    columns = ["QRS", "PQ Duration", "QRS Duration", "QTc Duration", "P Amplitude", "R Amplitude"]
    table_data = []

    
    for i, analysis in enumerate(ekg_analysis):
        row = [
            f"QRS {i+1}",
            f"{analysis['PQ Duration']['value']:.3f} s - {'Normalne' if analysis['PQ Duration']['normal'] else 'Nieprawidłowe'}",
            f"{analysis['QRS Duration']['value']:.3f} s - {'Normalne' if analysis['QRS Duration']['normal'] else 'Nieprawidłowe'}",
            f"{analysis['QTc Duration']['value']:.3f} s - {'Normalne' if analysis['QTc Duration']['normal'] else 'Nieprawidłowe'}",
            f"{analysis['P Amplitude']['value']:.2f} mV - {'Normalne' if analysis['P Amplitude']['normal'] else 'Nieprawidłowe'}",
            f"{analysis['R Amplitude']['value']:.2f} mV - {'Normalne' if analysis['R Amplitude']['normal'] else 'Nieprawidłowe'}",
        ]
        table_data.append(row)

    
    table = ax.table(cellText=table_data, colLabels=columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.2, 1.2)
    ax.set_title("Wyniki Analizy Parametrów EKG (QRS)", fontsize=12, fontweight="bold")








def main():
    parser = argparse.ArgumentParser(description="Symulacja Analizy Online EKG i Akcelerometru.")
    parser.add_argument("--ecg_path", type=str, required=True, help="Ścieżka do pliku EKG (.hea lub .json)")
    parser.add_argument("--sensor_path", type=str, required=True, help="Ścieżka do pliku akcelerometru (.hea lub .json)")
    parser.add_argument("--segment_duration", type=int, default=10, help="Długość segmentu analizy (s)")
    args = parser.parse_args()

    try:
        
        ecg_signal, ecg_fs, _ = load_data(args.ecg_path, "ecg")
        filtered_ecg = bandpass_filter(ecg_signal, ecg_fs)
        r_peaks = detect_qrs(filtered_ecg, ecg_fs)
        ekg_analysis = analyze_ekg_parameters(r_peaks, filtered_ecg, ecg_fs)

        
        sensor_signal, sensor_fs = load_data(args.sensor_path, "acc")

        
        simulate_online_analysis(ecg_signal, ecg_fs, sensor_signal, sensor_fs, segment_duration=args.segment_duration)

        
        display_ekg_analysis_table(ekg_analysis)

    except Exception as e:
        print(f"Błąd: {e}")


if __name__ == "__main__":
    main()