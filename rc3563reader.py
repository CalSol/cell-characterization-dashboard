import kivy
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
from kivy.uix.recycleview import RecycleView
from kivy.uix.recycleview.views import RecycleDataViewBehavior
from kivy.uix.gridlayout import GridLayout
from kivy.uix.behaviors import FocusBehavior
from kivy.uix.recycleboxlayout import RecycleBoxLayout
from kivy.uix.recyclegridlayout import RecycleGridLayout
from kivy.uix.behaviors import CompoundSelectionBehavior
from kivy.properties import BooleanProperty, StringProperty, ObjectProperty
from kivy.clock import Clock
import serial
import struct
import pandas as pd
import threading

class SelectableRecycleGridLayout(FocusBehavior, CompoundSelectionBehavior, RecycleGridLayout):
    pass

class SelectableLabel(RecycleDataViewBehavior, Label):
    index = None
    selected = BooleanProperty(False)
    selectable = BooleanProperty(True)

    def refresh_view_attrs(self, rv, index, data):
        self.index = index
        return super(SelectableLabel, self).refresh_view_attrs(rv, index, data)

    def on_touch_down(self, touch):
        if super(SelectableLabel, self).on_touch_down(touch):
            return True
        if self.collide_point(*touch.pos) and self.selectable:
            return self.parent.select_with_touch(self.index, touch)

    def apply_selection(self, rv, index, is_selected):
        self.selected = is_selected

class RV(RecycleView):
    def __init__(self, **kwargs):
        super(RV, self).__init__(**kwargs)
        self.data = [{'text': str(x)} for x in range(100)]

class SerialReaderApp(App):
    def build(self):
        self.data = []  # Store recorded data
        self.serial_port = None
        self.running = False
        self.last_values = (None, None)
        self.timer = None
        
        layout = BoxLayout(orientation='vertical')
        
        self.resistance_label = Label(text='Resistance: --')
        self.voltage_label = Label(text='Voltage: --')
        
        self.port_input = TextInput(hint_text='Enter Serial Port (e.g., COM3 or /dev/ttyUSB0)')
        self.connect_button = Button(text='Connect')
        self.connect_button.bind(on_press=self.connect_serial)
        
        self.record_button = Button(text='Record')
        self.record_button.bind(on_press=self.record_reading)
        
        self.export_button = Button(text='Export to CSV')
        self.export_button.bind(on_press=self.export_to_csv)
        
        self.rv = RV()
        layout.add_widget(self.port_input)
        layout.add_widget(self.connect_button)
        layout.add_widget(self.resistance_label)
        layout.add_widget(self.voltage_label)
        layout.add_widget(self.record_button)
        layout.add_widget(self.export_button)
        layout.add_widget(self.rv)
        
        return layout
    
    def connect_serial(self, instance):
        port = self.port_input.text.strip()
        if port:
            try:
                self.serial_port = serial.Serial(port, 115200, timeout=1)
                self.running = True
                self.read_thread = threading.Thread(target=self.read_serial)
                self.read_thread.start()
                self.connect_button.text = 'Connected'
                self.connect_button.disabled = True
            except Exception as e:
                self.resistance_label.text = f'Error: {str(e)}'
    
    def read_serial(self):
        while self.running:
            try:
                pkt = self.serial_port.read(10)
                if len(pkt) == 10:
                    status_disp, r_range_code, r_disp, sign_code, v_range_code, v_disp = struct.unpack('BB3s BB3s', pkt)
                    
                    r_disp = struct.unpack('I', r_disp + b'\x00')[0]
                    resistance = float(r_disp) / 1e4
                    
                    v_disp = struct.unpack('I', v_disp + b'\x00')[0]
                    voltage = float(v_disp) / 1e4 if sign_code == 1 else -float(v_disp) / 1e4
                    
                    self.resistance_label.text = f'Resistance: {resistance} mΩ'
                    self.voltage_label.text = f'Voltage: {voltage} V'
                    
                    if self.last_values == (resistance, voltage):
                        if self.timer is None:
                            self.timer = Clock.schedule_once(lambda dt: self.record_reading(None), 2)
                    else:
                        if self.timer is not None:
                            self.timer.cancel()
                            self.timer = None
                        self.last_values = (resistance, voltage)
            except Exception as e:
                self.resistance_label.text = f'Error: {str(e)}'
    
    def record_reading(self, instance):
        resistance_text = self.resistance_label.text.replace('Resistance: ', '').replace(' mΩ', '')
        voltage_text = self.voltage_label.text.replace('Voltage: ', '').replace(' V', '')
        
        try:
            resistance = float(resistance_text)
            voltage = float(voltage_text)
            self.data.append({'Resistance (mΩ)': resistance, 'Voltage (V)': voltage})
            self.update_rv()
        except ValueError:
            pass  # Ignore invalid readings
    
    def update_rv(self):
        self.rv.data = [{'text': f"{row['Resistance (mΩ)']}, {row['Voltage (V)']}"} for row in self.data]
    
    def export_to_csv(self, instance):
        if self.data:
            df = pd.DataFrame(self.data)
            df.to_csv('readings.csv', mode='a', index=False, header=False)
            self.data.clear()  # Clear the dataframe after appending
            self.resistance_label.text = 'Data appended to readings.csv'
            self.update_rv()
        else:
            self.resistance_label.text = 'No data to export'
    
    def on_stop(self):
        self.running = False
        if self.serial_port:
            self.serial_port.close()


if __name__ == '__main__':
    SerialReaderApp().run()