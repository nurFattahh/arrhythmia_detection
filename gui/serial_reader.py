from PyQt5.QtCore import QThread, pyqtSignal
import serial
import time

try:
    from pyshimmer import ShimmerBluetooth, DEFAULT_BAUDRATE, DataPacket, EChannelType
    SHIMMER_AVAILABLE = True
except ImportError:
    SHIMMER_AVAILABLE = False
    print("⚠️ pyshimmer not found. Shimmer mode will be disabled.")

class ShimmerReader(QThread):
    """Thread untuk membaca data dari Shimmer ECG"""
    new_data = pyqtSignal(float)
    error_signal = pyqtSignal(str)

    def __init__(self, port, baudrate=DEFAULT_BAUDRATE if SHIMMER_AVAILABLE else 115200):
        super().__init__()
        self.port = port
        self.baudrate = baudrate
        self._running = True
        self.shim_dev = None
    
    def adc_to_millivolts(self, adc_signal, gain=6, offset=0, vref=2.42,adc_bits=24):
        adc_sensitivity = (vref * 1000) / (2 ** (adc_bits - 1) - 1)
        return ((adc_signal - offset) * adc_sensitivity)/gain

    def handle_packet(self, pkt: DataPacket):
        """Callback untuk setiap paket data dari Shimmer"""
        try:
            # Ambil data ECG dari channel CH2 (LL-LA)
            cur_value = pkt[EChannelType.EXG_ADS1292R_1_CH1_24BIT]
            # Konversi ke mV (sesuaikan gain jika perlu)
            value_mv = ShimmerReader.adc_to_millivolts(self, cur_value, gain=6, offset=0)
            print(f"Amplitudo: {value_mv:.2f} mV")
            self.new_data.emit(value_mv)
        except KeyError:
            # Channel belum tersedia
            pass
        except Exception as e:
            print(f"Error handling packet: {e}")

    def run(self):
        try:
            # Buka serial connection
            serial_conn = serial.Serial(self.port, self.baudrate)
            
            # Inisialisasi Shimmer
            self.shim_dev = ShimmerBluetooth(serial_conn)
            self.shim_dev.initialize()
            
            # Get device name untuk konfirmasi
            dev_name = self.shim_dev.get_device_name()
            print(f"✅ Connected to Shimmer: {dev_name}")
            
            # Register callback
            self.shim_dev.add_stream_callback(self.handle_packet)
            
            # Start streaming
            self.shim_dev.start_streaming()
            
            # Keep thread alive while streaming
            while self._running:
                time.sleep(0.1)
            
            # Cleanup
            self.shim_dev.stop_streaming()
            self.shim_dev.shutdown()
            serial_conn.close()
            
        except Exception as e:
            error_msg = f"Shimmer error: {str(e)}"
            print(error_msg)
            self.error_signal.emit(error_msg)

    def stop(self):
        self._running = False

class SerialReader(QThread):
    new_data = pyqtSignal(float)

    def __init__(self, port, baudrate=115200):
        super().__init__()
        self.port = port
        self.baudrate = baudrate
        self._running = True
        self.ser = None

    def run(self):
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=0.5)
            while self._running:
                if not self.ser or not self.ser.is_open:
                    break
                line = self.ser.readline().decode(errors="ignore").strip()
                if line:
                    try:
                        val = float(line)
                        self.new_data.emit(val)
                    except:
                        pass
            # keluar loop, tutup port
            if self.ser and self.ser.is_open:
                try:
                    self.ser.close()
                except:
                    pass
        except Exception as e:
            print("Serial error:", e)

    def stop(self):
        self._running = False
        # paksa close port untuk hentikan blocking read
        try:
            if self.ser and self.ser.is_open:
                self.ser.close()
        except:
            pass
