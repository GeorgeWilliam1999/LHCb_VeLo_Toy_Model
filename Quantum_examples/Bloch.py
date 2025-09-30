# If needed:
# !pip install qiskit qiskit-visualization ipywidgets


import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.visualization import plot_bloch_multivector
import ipywidgets as widgets
from ipywidgets import FloatSlider, Tab, VBox, Output
from IPython.display import display, clear_output

def state_from_theta_phi(theta, phi):
    a = np.cos(theta/2.0)
    b = np.exp(1j*phi) * np.sin(theta/2.0)
    return Statevector([a, b])

def state_from_rx_ry_rz(rx, ry, rz):
    qc = QuantumCircuit(1)
    qc.rx(rx, 0); qc.ry(ry, 0); qc.rz(rz, 0)
    return Statevector.from_instruction(qc), qc

# ----- Direct θ, φ -----
theta_slider = FloatSlider(value=np.pi/3, min=0, max=np.pi, step=0.01, description='θ')
phi_slider   = FloatSlider(value=np.pi/4, min=0, max=2*np.pi, step=0.01, description='φ')
out_direct = Output()

def update_direct(_=None):
    with out_direct:
        clear_output(wait=True)
        st = state_from_theta_phi(theta_slider.value, phi_slider.value)
        print(f"Statevector ≈ [{st.data[0]:.3f}, {st.data[1]:.3f}]")
        fig = plot_bloch_multivector(st)
        display(fig)

theta_slider.observe(update_direct, names='value')
phi_slider.observe(update_direct, names='value')
direct_tab = VBox([theta_slider, phi_slider, out_direct])

# ----- Rx/Ry/Rz -----
rx = FloatSlider(value=0.0,   min=-2*np.pi, max=2*np.pi, step=0.01, description='Rx')
ry = FloatSlider(value=np.pi/2, min=-2*np.pi, max=2*np.pi, step=0.01, description='Ry')
rz = FloatSlider(value=0.0,   min=-2*np.pi, max=2*np.pi, step=0.01, description='Rz')
out_gate = Output()

def update_gate(_=None):
    with out_gate:
        clear_output(wait=True)
        st, qc = state_from_rx_ry_rz(rx.value, ry.value, rz.value)
        print(f"Statevector ≈ [{st.data[0]:.3f}, {st.data[1]:.3f}]")
        try:
            display(qc.draw(output='mpl'))
        except Exception:
            print(qc.draw())  # text fallback
        fig = plot_bloch_multivector(st)
        display(fig)

for s in (rx, ry, rz):
    s.observe(update_gate, names='value')

gate_tab = VBox([rx, ry, rz, out_gate])

tabs = Tab(children=[direct_tab, gate_tab])
tabs.set_title(0, 'Direct θ, φ')
tabs.set_title(1, 'Rx/Ry/Rz')

display(tabs)
update_direct()
update_gate()
