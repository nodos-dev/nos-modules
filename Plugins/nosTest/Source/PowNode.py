import struct
import nodos as nos
from typing import Any

class PyPow(nos.Node):
    def __init__(self):
        self.previous_base = 0
        self.previous_exponent = 0

    def on_pin_value_changed(self, args: nos.PinValueChangedArgs):
        base_mem: memoryview = args.pin_value

        if(args.pin_name == "Base"):
            nos.log_info(f"Pin {args.pin_name} changed from {self.previous_base} to {struct.unpack('f', base_mem.cast('B'))[0]}")
            self.previous_base = struct.unpack('f', base_mem.cast('B'))[0]
        elif(args.pin_name == "Exponent"):
            nos.log_info(f"Pin {args.pin_name} changed from {self.previous_exponent} to {struct.unpack('f', base_mem.cast('B'))[0]}")
            self.previous_exponent = struct.unpack('f', base_mem.cast('B'))[0]

    def execute_node(self, args: nos.NodeExecuteArgs):
        base_mem: memoryview = args.get_pin_value("Base")
        exp_mem: memoryview = args.get_pin_value("Exponent")
        base = struct.unpack('f', base_mem.cast('B'))[0]
        exp = struct.unpack('f', exp_mem.cast('B'))[0]
        # You can also directly set the memory if your type does not 
        # need custom handling & you are inside Nodos Scheduler thread.
        # struct.pack_into('f', out_mem, 0, base ** exp)
        result_buf = struct.pack('f', base ** exp)
        nos.set_pin_value(args.get_pin_id("Result"), result_buf)
        return nos.result.SUCCESS
    
    def on_node_created(self, args: Any):
        print("Node created")
    
    def on_pin_connected(self, args: nos.PinConnectedArgs):
        print("Pin connected")
    
    def on_pin_disconnected(self, args: nos.PinDisconnectedArgs):
        print("Pin disconnected")